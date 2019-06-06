"""
Soft Spectral Normalization (not enforced, only <= coeff) for Conv2D layers
Based on: Regularisation of Neural Networks by Enforcing Lipschitz Continuity
    (Gouk et al. 2018)
    https://arxiv.org/abs/1804.04368
"""
import torch
from torch.nn.functional import normalize, conv_transpose2d, conv2d
from torch.nn.parameter import Parameter
from torch.nn import ConvTranspose2d


class SpectralNormConv(object):
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.

    def __init__(self, coeff, input_dim, name='weight', n_power_iterations=1, eps=1e-12):
        self.coeff = coeff
        self.input_dim = input_dim
        self.name = name
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps


    def compute_weight(self, module, do_power_iteration):
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important bahaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is alreay on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        sigma_log = getattr(module, self.name + '_sigma') # for logging

        # get settings from conv-module (for transposed convolution)
        stride = module.stride
        padding = module.padding

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v_s = conv_transpose2d(u.view(self.out_shape), weight, stride=stride,
                                         padding=padding, output_padding=0)
                    # Note: out flag for in-place changes
                    v = normalize(v_s.view(-1), dim=0, eps=self.eps, out=v)
                    
                    u_s = conv2d(v.view(self.input_dim), weight, stride=stride, padding=padding,
                           bias=None)
                    u = normalize(u_s.view(-1), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone()
                    v = v.clone()
        weight_v = conv2d(v.view(self.input_dim), weight, stride=stride, padding=padding,
                           bias=None)
        weight_v = weight_v.view(-1)
        sigma = torch.dot(u.view(-1), weight_v)  
        # enforce spectral norm only as constraint
        factorReverse = torch.max(torch.ones(1).to(weight.device),
                                  sigma / self.coeff)
        # for logging
        weight_v_det = weight_v.detach()
        u_det = u.detach()
        torch.max(torch.dot(u_det.view(-1), weight_v_det),
                  torch.dot(u_det.view(-1), weight_v_det), out=sigma_log)

        # rescaling
        weight = weight / (factorReverse + 1e-5)  # for stability
        return weight

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))


    @staticmethod
    def apply(module, coeff, input_dim, name, n_power_iterations, eps):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNormConv) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNormConv(coeff, input_dim, name, n_power_iterations, eps)
        weight = module._parameters[name]

        with torch.no_grad():
            num_input_dim = input_dim[0]* input_dim[1]* input_dim[2]* input_dim[3]
            v = normalize(torch.randn(num_input_dim), dim=0, eps=fn.eps)

            # get settings from conv-module (for transposed convolution)
            stride = module.stride
            padding = module.padding
            # forward call to infer the shape
            u = conv2d(v.view(input_dim), weight, stride=stride, padding=padding,
                               bias=None)
            fn.out_shape = u.shape
            num_output_dim = fn.out_shape[0]* fn.out_shape[1]* fn.out_shape[2]* fn.out_shape[3]
            # overwrite u with random init
            u = normalize(torch.randn(num_output_dim), dim=0, eps=fn.eps)



        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1).to(weight.device))

        module.register_forward_pre_hook(fn)

        module._register_state_dict_hook(SpectralNormConvStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormConvLoadStateDictPreHook(fn))
        return fn



class SpectralNormConvLoadStateDictPreHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        fn = self.fn
        version = local_metadata.get('spectral_norm_conv', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_orig = state_dict[prefix + fn.name + '_orig']
                weight = state_dict.pop(prefix + fn.name)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[prefix + fn.name + '_u']


class SpectralNormConvStateDictHook(object):
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm_conv' not in local_metadata:
            local_metadata['spectral_norm_conv'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm_conv']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm_conv']: {}".format(key))
        local_metadata['spectral_norm_conv'][key] = self.fn._version


def spectral_norm_conv(module, coeff, input_dim, name='weight', n_power_iterations=1, eps=1e-12):
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
         \mathbf{W} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})} \\
         \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generaive Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectal norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is 0, except for modules that are instances of
            ConvTranspose1/2/3d, when it is 1
    Returns:
        The original module with the spectal norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        Linear (20 -> 40)
        >>> m.weight_u.size()
        torch.Size([20])
    """
    input_dim_4d = (1, input_dim[0], input_dim[1], input_dim[2])
    SpectralNormConv.apply(module, coeff, input_dim_4d, name, n_power_iterations, eps)
    return module


def remove_spectral_norm_conv(module, name='weight'):
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNormConv) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, module))
