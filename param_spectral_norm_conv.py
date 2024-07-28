"""
Obtained from torch.nn.utils.parametrizations at tag 2.3
This version is modified to apply a soft spectral normalization
Spectral Normalization from https://arxiv.org/abs/1802.05957
Soft Spectral Normalization (not enforced, only <= coeff) for Conv2D layers
Based on: Regularisation of Neural Networks by Enforcing Lipschitz Continuity
    (Gouk et al. 2018)
    https://arxiv.org/abs/1804.04368
"""
import torch
from typing import Any, Optional, TypeVar
from torch.nn.modules import Module


from enum import Enum, auto

import torch
from torch import Tensor
from torch.nn.utils import parametrize
from torch.nn.modules import Module
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

from typing import Optional


class _SpectralNormConv(Module):
    def __init__(
        self,
        weight: torch.Tensor,
        input_dim: torch.Size,
        stride: _size_2_t,
        padding: _size_2_t,
        n_power_iterations: int = 1,
        eps: float = 1e-12,
        coeff:float = 1.0,
    ) -> None:
        self.coeff = coeff
        self.input_dim = input_dim
        self.stride = stride
        self.padding = padding
        self.n_power_iterations = n_power_iterations
        super().__init__()
        ndim = weight.ndim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.dim = ndim
        self.eps = eps
        if ndim > 1:
            with torch.no_grad():
                num_input_dim = input_dim[0] * input_dim[1] * input_dim[2] * input_dim[3]
                v = F.normalize(torch.randn(num_input_dim), dim=0, eps=self.eps)

                # get settings from conv-module (for transposed convolution)
                stride = self.stride
                padding = self.padding
                # forward call to infer the shape
                u = F.conv2d(v.view(input_dim), weight, stride=stride, padding=padding, bias=None)
                self.out_shape = u.shape
                num_output_dim = self.out_shape[0] * self.out_shape[1] * self.out_shape[2] * self.out_shape[3]
                # overwrite u with random init
                u = F.normalize(torch.randn(num_output_dim), dim=0, eps=self.eps)

            self.register_buffer('_u', u)
            self.register_buffer('_v', v)

            # Start with u, v initialized to some reasonable values by performing a number
            # of iterations of the power method
            self._power_method(weight, 15)

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # Precondition
        assert weight.ndim > 1

        if self.dim != 0:
            # permute dim to front
            weight = weight.permute(self.dim, *(d for d in range(weight.dim()) if d != self.dim))

        return weight.flatten(1)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:
        # Precondition
        assert weight_mat.ndim > 1

        output_padding = 0
        # NOTE:difference
        if self.stride[0] > 1:
            # Note: the below does not generalize to stride > 2
            output_padding = 1 - self.input_dim[-1] % 2

        for _ in range(n_power_iterations):
            # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
            # are the first left and right singular vectors.
            # This power iteration produces approximations of `u` and `v`.
            # In convolutions we can leverage the fact that:
            # Conv2d(v, W) simulates multiplying W by v.
            # ConvTranspose2d(u, W) simulates multiplying W^T by u.

            v_s = F.conv_transpose2d(
                self._u.view(self.out_shape),
                weight_mat,
                stride=self.stride,
                padding=self.padding,
                output_padding=output_padding,
            )
            self._v = F.normalize(v_s.view(-1),
                                  dim=0, eps=self.eps, out=self._v)   # type: ignore[has-type]
            
            u_s = F.conv2d(self._v.view(self.input_dim),
                            weight_mat, 
                            stride=self.stride, 
                            padding=self.padding, 
                            bias=None,)        
            self._u = F.normalize(u_s.view(-1),      # type: ignore[has-type]
                                  dim=0, eps=self.eps, out=self._u)   # type: ignore[has-type]
            
    @torch.autocast(device_type="cuda",enabled=False)
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            # weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            with torch.no_grad():
                weight_v = F.conv2d(v.view(self.input_dim), weight, stride=self.stride, padding=self.padding, bias=None)
                weight_v = weight_v.view(-1)
                sigma = torch.dot(u.view(-1), weight_v)
                # soft normalization: only when sigma larger than coeff
                factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
            return weight / (factor + 1e-5)

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # we may want to assert here that the passed value already
        # satisfies constraints
        return value


def spectral_norm_conv(module: Module,
                  coeff: float,
                  input_dim: tuple,
                  n_power_iterations: int = 1,
                  name: str = 'weight',
                  eps: float = 1e-12) -> Module:

    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )
    input_dim_4d = torch.Size([1, int(input_dim[0]), int(input_dim[1]), int(input_dim[2])])
    parametrize.register_parametrization(module, name, _SpectralNormConv(weight,input_dim_4d,module.stride, module.padding, n_power_iterations, eps, coeff))
    return module
