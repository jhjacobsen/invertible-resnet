"""
Code for "i-RevNet: Deep Invertible Networks"
https://openreview.net/pdf?id=HJsjkMb0Z
ICLR 2018
"""

import torch
import torch.nn as nn

from torch.nn import Parameter


def split(x):
    n = int(x.size(1)/2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class Split(nn.Module):
    def __init__(self):
        super(Split, self).__init__()

    def forward(self, x):
        n = int(x.size(1) / 2)
        x1 = x[:, :n, :, :].contiguous()
        x2 = x[:, n:, :, :].contiguous()
        return x1, x2

    def inverse(self, x1, x2):
        return torch.cat((x1, x2), 1)



class squeeze(nn.Module):
    def __init__(self, block_size):
        super(squeeze, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def inverse(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.contiguous().view(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()


class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))


def get_all_params(var, all_params):
    if isinstance(var, Parameter):
        all_params[id(var)] = var.nelement()
    elif hasattr(var, "creator") and var.creator is not None:
        if var.creator.previous_functions is not None:
            for j in var.creator.previous_functions:
                get_all_params(j[0], all_params)
    elif hasattr(var, "previous_functions"):
        for j in var.previous_functions:
            get_all_params(j[0], all_params)


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective, z_list, labels=None):
        raise NotImplementedError

    def reverse_(self, y, objective, labels=None):
        raise NotImplementedError


class ActNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.Tensor(num_channels))
        self._shift = Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :]

    def shift(self):
        return self._shift[None, :]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() 
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())


class ActNorm2D(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ActNorm2D, self).__init__()
        self.eps = eps
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.Tensor(num_channels))
        self._shift = Parameter(torch.Tensor(num_channels))
        self._init = False

    def log_scale(self):
        return self._log_scale[None, :, None, None]

    def shift(self):
        return self._shift[None, :, None, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)
                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                zero_mean = x - mean[None, :, None, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                self._shift.data = - mean * torch.exp(log_scale)
                self._log_scale.data = log_scale
                self._init = True
        log_scale = self.log_scale()
        logdet = log_scale.sum() * x.size(2) * x.size(3)
        return x * torch.exp(log_scale) + self.shift(), logdet

    def inverse(self, x):
        return (x - self.shift()) * torch.exp(-self.log_scale())


class MaxMinGroup(nn.Module):
    def __init__(self, group_size, axis=-1):
        super(MaxMinGroup, self).__init__()
        self.group_size = group_size
        self.axis = axis

    def forward(self, x):
        maxes = maxout_by_group(x, self.group_size, self.axis)
        mins = minout_by_group(x, self.group_size, self.axis)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'group_size: {}'.format(self.group_size)
	
def process_maxmin_groupsize(x, group_size, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % group_size:
        raise ValueError('number of features({}) is not a '
                         'multiple of group_size({})'.format(num_channels, num_channels))
    size[axis] = -1
    if axis == -1:
        size += [group_size]
    else:
        size.insert(axis+1, group_size)
    return size


def maxout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout_by_group(x, group_size, axis=-1):
    size = process_maxmin_groupsize(x, group_size, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]


if __name__ == "__main__":
    batch_size = 13
    num_channels = 3
    h = 32
    w = 32
    x = torch.randn((batch_size, num_channels, h, w))

    AN = ActNorm2D(num_channels)

    out1, _ = AN(x)
    x_re = AN.inverse(out1)

    print((x - x_re).abs().mean())
    out2, _ = AN(x)
    s = torch.transpose(out2, 0, 1).contiguous().view(num_channels, -1).std(dim=1)
    m = torch.transpose(out2, 0, 1).contiguous().view(num_channels, -1).mean(dim=1)
    print(s, m)
