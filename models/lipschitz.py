import math
from torch._six import container_abcs
from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class SpectralNormLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, coeff=0.97, n_iterations=1):
        super(SpectralNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        h, w = self.weight.shape
        self.register_buffer('u', F.normalize(self.weight.new_empty(h).normal_(0, 1), dim=0))
        self.register_buffer('v', F.normalize(self.weight.new_empty(w).normal_(0, 1), dim=0))
        self.compute_weight(True, 200)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=False, n_iterations=None):
        n_iterations = n_iterations if n_iterations is not None else self.n_iterations
        u = self.u
        v = self.v
        weight = self.weight
        if update:
            with torch.no_grad():
                for _ in range(self.n_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                if self.n_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        weight = self.compute_weight(update=self.training)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class BjorckLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, coeff=0.97, n_iterations=1):
        super(BjorckLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.compute_weight(True, 3)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=False, n_iterations=None):
        n_iterations = n_iterations if n_iterations is not None else self.n_iterations

        with torch.set_grad_enabled(not update):
            A = self.weight
            m, n = A.shape
            pmax = torch.max(torch.abs(A)) * (n * m)**0.5
            p1 = torch.max(torch.sum(torch.abs(A), 0)) * n**0.5
            pinf = torch.max(torch.sum(torch.abs(A), 1)) * m**0.5
            max_sing = min(pmax, p1, pinf)
            A = A / max_sing if max_sing > 1 else A
            for _ in range(n_iterations):
                A = A + 0.5 * A - 0.5 * torch.mm(A, torch.mm(A.transpose(0, 1), A))

        if update:
            self.weight.data.copy_(A.detach() * self.coeff)
            return self.weight
        else:
            return A * self.coeff

    def forward(self, input):
        weight = self.compute_weight(update=not self.training)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ProjectedLinear(nn.Module):
    """Projects the weights s.t. ||W||_F <= 1 where ||W||_2 <= ||W||_F."""

    def __init__(self, in_features, out_features, bias=True, coeff=0.97, n_iterations=None):
        super(ProjectedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.compute_weight(True, None)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=False, n_iterations=None):
        fro = torch.norm(self.weight)
        # soft normalization
        factor = torch.max(torch.ones(1).to(self.weight.device), fro / self.coeff)
        weight = self.weight / factor
        if update:
            self.weight.data.copy_(weight)
            return self.weight
        else:
            return weight

    def forward(self, input):
        weight = self.compute_weight(update=False)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class ProjectedConv2d(nn.Module):
    """Projects the weights s.t. ||W||_F <= 1 where ||W||_2 <= ||W||_F."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, coeff=0.97):
        super(ProjectedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.coeff = coeff
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.compute_weight(update=True)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=False, spatial_dims=(1, 1)):
        h, w = spatial_dims
        fro = torch.norm(self.weight) * math.sqrt(h * w)
        # soft normalization
        factor = torch.max(torch.ones(1).to(self.weight.device), fro / self.coeff)
        weight = self.weight / factor
        if update:
            self.weight.data.copy_(weight)
            return self.weight
        else:
            return weight

    def forward(self, input):
        weight = self.compute_weight(update=False, spatial_dims=input.shape[2:4])
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, 1, 1)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}' ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
