import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from models.invertible_layers import *
from models.utils import gaussian_diag, logmeanexp, indexes_to_one_hot, softmax
from copy import copy
import pdb

class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward_(self, x, objective, z_list, labels=None):
        raise NotImplementedError

    def reverse_(self, y, objective, labels=None):
        raise NotImplementedError

class FCZeroInit(nn.Module):
    def __init__(self, input_shape):
        super(FCZeroInit,self).__init__()
        self.shape = list(input_shape)
        self.param = nn.Parameter(torch.zeros(self.shape[0],np.prod(self.shape[1:])))
    def forward(self,x):
        x = torch.matmul(x,self.param).view(x.size()[0],self.shape[1],self.shape[2],self.shape[3])
        return  x


class MeanVarFC(nn.Module):
    def __init__(self, input_shape):
        super(MeanVarFC,self).__init__()
        shape = list(input_shape)
        shape[0] = 1
        shape[1] *= 2
        self.param = nn.Parameter(0.01*torch.randn(shape))
    def forward(self,x):
        x = x+self.param
        return  x


class Conv2dZeroInit(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=0, logscale=3.):
        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=padding)
        self.register_parameter("logs", nn.Parameter(torch.zeros(channels_out, 1, 1)))
        self.logscale_factor = logscale

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        out = super().forward(input)
        return out * torch.exp(self.logs * self.logscale_factor)


class IdentityFunc(nn.Module):
    def __init__(self):
        super(IdentityFunc,self).__init__()
    def forward(self,x):
        return x


def compute_distributions(p_x_given_class, labels, n_classes, ycond=False):
    p_x = logmeanexp(p_x_given_class,dim=1)
    p_class_given_x = softmax(p_x_given_class)
    if ycond:
        labels_onehot = indexes_to_one_hot(labels.cpu(),n_classes).byte().cuda()
        p_x_given_y = torch.masked_select(p_x_given_class, labels_onehot)
        p_y_given_x = torch.masked_select(p_class_given_x, labels_onehot)
    else:
        assert(p_x_given_class.shape[1] == 1)
        p_x_given_y = p_x_given_class[:,0]
        p_y_given_x = torch.ones_like(p_x)
    pred = p_x_given_class.argmax(dim=1)
    if not ycond:
        pred -= 1
    correct =  pred.eq(labels.view_as(pred)).float().mean()
    return p_x, p_y_given_x, p_x_given_y, pred, correct


class PriorClass(Layer):
    def __init__(self, input_shape, learntop=False, conv_prior=False, ycond=False, n_classes=10):
        super(PriorClass, self).__init__()
        self.input_shape = input_shape
        self.learntop = learntop
        self.conv_prior = conv_prior
        self.ycond = ycond
        if not ycond:
            self.n_classes = 1
        else:
            self.n_classes = n_classes


class GaussianPriorBasic(nn.Module):

    def __init__(self, input_shape):
        super(GaussianPriorBasic, self).__init__()
        self.input_shape = input_shape

    def forward(self, x, objective):
        mean_and_logsd = torch.cat([torch.zeros_like(x) for _ in range(2)], dim=1)
        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        pz = self._gaussian_diag(mean, logsd)
        objective += pz.logp(x)
        return x, objective

    def _flatten_sum(self, logps):
        while len(logps.size()) > 1: 
            logps = logps.sum(dim=-1)
        return logps

    def _gaussian_diag(self, mean, logsd):
        class o(object):
            
            pass
    
            def logps(x):
                return  -0.5 * (2. * logsd) - ((x - mean) ** 2) / (2* torch.exp(logsd))
    
            def sample():

                eps = torch.zeros_like(mean).normal_()
                return mean + torch.exp(logsd) * eps
            
            def logp(x):
                dim = float(np.prod(x.shape[1:]))
                Log2PIdim = -0.5* (dim*float(np.log(2 * np.pi)))
                return Log2PIdim + self._flatten_sum(o.logps(x))

        return o
    
    def reverse_(self, batch_size):
        bs, c, h, w = self.input_shape
        bs = batch_size
        mean_and_logsd = torch.cuda.FloatTensor(bs, 2 * c, h, w).fill_(0.)
        mean, logsd = torch.chunk(mean_and_logsd, 2, dim=1)
        pz = self._gaussian_diag(mean, logsd)
        z = pz.sample().view(bs, c, h, w).cuda()
        return z

class GaussianPrior(nn.Module):
    def __init__(self, input_shape, learntop=False, conv_prior=False, ycond=False, n_classes=1):
        super(GaussianPrior, self).__init__()
        self.learntop = learntop
        self.input_shape = input_shape
        self.conv_prior = conv_prior
        self.ycond = ycond
        self.n_classes = n_classes
        if self.learntop and self.conv_prior:
            self.prior_f = nn.ModuleList([Conv2dZeroInit(2 * input_shape[1], 2 * input_shape[1], 3, padding=(3 - 1) // 2) for i in range(self.n_classes)])
        elif self.learntop:
            self.prior_f = nn.ModuleList([MeanVarFC(input_shape) for i in range(self.n_classes)])
        else:
            self.prior_f = nn.ModuleList([IdentityFunc() for i in range(self.n_classes)])
        self.softplus = nn.Softplus()
        self.shape = input_shape

    def forward(self, x, z_list, labels=None):

        means = []
        logsd  = []
        pzs = []
        p_x_given_class = torch.zeros(x.shape[0],self.n_classes).cuda()
        for i in range(self.n_classes):
            mean_and_logsd = torch.cat([torch.zeros_like(x) for _ in range(2)], dim=1)
            mean_and_logsd = self.prior_f[i](mean_and_logsd)
            mean_, logsd_ = torch.chunk(mean_and_logsd, 2, dim=1)
            if self.learntop:
                min_logsd = -1.
                logsd_ = min_logsd+self.softplus(logsd_)
            means.append(mean_); logsd.append(logsd_)

            pzs.append(gaussian_diag(mean_, logsd_))
            p_x_given_class[:,i] = pzs[-1].logp(x)
        return p_x_given_class, z_list

    def reverse(self, batch_size, labels=None):
        shape = list(self.input_shape)
        shape[0] = batch_size
        samples_ = torch.zeros(shape+[self.n_classes])
        for i in range(self.n_classes):
            mean_and_logsd = torch.cat([torch.zeros(shape) for _ in range(2)], dim=1).cuda()
            mean_and_logsd = self.prior_f[i](mean_and_logsd)
            mean_, logsd_ = torch.chunk(mean_and_logsd, 2, dim=1)
            if self.learntop:
                min_logsd = -1.
                logsd_ = min_logsd+self.softplus(logsd_)

            pz = gaussian_diag(mean_, logsd_)
            samples_[:,:,:,:,i] = pz.sample()
        if self.n_classes>1:
            if labels is None:
                labels = np.random.randint(low=0,high=self.n_classes,size=batch_size)
            labels_ = labels.cpu().view(-1, 1, 1, 1, 1).repeat(1, shape[1], shape[2], shape[3], 1)
            samples_ = torch.gather(samples_, 4, labels_)
        samples = samples_.view(shape).cuda()
        return samples

