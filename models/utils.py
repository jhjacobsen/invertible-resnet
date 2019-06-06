import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.nn.modules.batchnorm import _BatchNorm

import numpy as np
import pdb
import os

# ------------------------------------------------------------------------------
# Utility Methods
# ------------------------------------------------------------------------------

def flatten_sum(logps):
    while len(logps.size()) > 1: 
        logps = logps.sum(dim=-1)
    return logps

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

def save_session(model, optim, args, epoch):
    path = os.path.join(args.save_dir, str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)

    # save the model and optimizer state
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(optim.state_dict(), os.path.join(path, 'optim.pth'))
    print('Successfully saved model')

def load_session(model, optim, args):
    try: 
        start_epoch = int(args.load_dir.split('/')[-1])
        model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model.pth')))
        optim.load_state_dict(torch.load(os.path.join(args.load_dir, 'optim.pth')))
        print('Successfully loaded model')
    except Exception as e:
        print('Could not restore session properly')

    return model, optim, start_epoch


# ------------------------------------------------------------------------------
# Distributions
# ------------------------------------------------------------------------------

def standard_gaussian(shape):
    mean, logsd = [torch.cuda.FloatTensor(shape).fill_(0.) for _ in range(2)]
    return gaussian_diag(mean, logsd)

def gaussian_diag(mean, logsd):
    class o(object):
        Log2PI = float(np.log(2 * np.pi))
        pass

        def logps(x):
            return  -0.5 * (o.Log2PI + 2. * logsd + ((x - mean) ** 2) / torch.exp(2. * logsd))

        def sample():
            eps = torch.cuda.FloatTensor(mean.size()).normal_()
            return mean + torch.exp(logsd) * eps

    o.logp  = lambda x: flatten_sum(o.logps(x))
    return o

def laplace_diag(mu, log_b):
    '''
    log(p(x;mu,b)) = sum -|x_i-mu_i|/b
    :param mu: mean
    :param b: var is 2b^2
    '''
    class o(object):


        def logps(x):
            return  -1 * (torch.abs((x-mu)/torch.exp(log_b)) +log_b + float(np.log(2)))

        def sample():
            eps = 0.0000001
            unif = torch.clamp(torch.rand(size=mu.shape) - .5, min=-0.5+eps, max=0.5-eps).cuda()
            samples = mu - torch.exp(log_b) * torch.sign(unif) * torch.log(1 - 2. * torch.abs(unif))
            return samples

    o.logp    = lambda x: flatten_sum(o.logps(x))
    return o

def cauchy_diag():

    class o(object):
        def logps(x):
            return -1.*(float(np.log(np.pi)) + torch.log(1+x**2))

    o.logp    = lambda x: flatten_sum(o.logps(x))
    return o


def indexes_to_one_hot(indexes, n_classes=None):
    """Converts a vector of indexes to a batch of one-hot vectors. """
    indexes = indexes.type(torch.int64).view(-1, 1)
    n_classes = n_classes if n_classes is not None else int(torch.max(indexes)) + 1
    if indexes.is_cuda:
      one_hots = torch.zeros(indexes.size()[0], n_classes).cuda().scatter_(1, indexes, 1)
    else:
      one_hots = torch.zeros(indexes.size()[0], n_classes).scatter_(1, indexes, 1)
    #one_hots = one_hots.view(*indexes.shape, -1)
    return one_hots

def logmeanexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().mean(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

softmax = nn.Softmax(dim=1)

def compute_distributions(p_x_given_class, labels, n_classes, ycond=False):
    #import ipdb; ipdb.set_trace()
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
