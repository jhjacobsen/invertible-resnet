"""
Code for "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""

import numpy as np
import torch
from scipy.linalg import logm
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn

def exact_matrix_logarithm_trace(Fx, x):
    """
    Computes slow-ass Tr(Ln(d(Fx)/dx))
    :param Fx: output of f(x)
    :param x: input
    :return: Tr(Ln(I + df/dx))
    """
    bs = Fx.size(0)
    outVector = torch.sum(Fx, 0).view(-1)
    outdim = outVector.size()[0]
    indim = x.view(bs, -1).size()
    jac = torch.empty([bs, outdim, indim[1]], dtype=torch.float)
    # for each output Fx[i] compute d(Fx[i])/d(x)
    for i in range(outdim):
        zero_gradients(x)
        jac[:, i, :] = torch.autograd.grad(outVector[i], x,
                                           retain_graph=True)[0].view(bs, outdim)
    jac = jac.cpu().numpy()
    iden = np.eye(jac.shape[1])
    log_jac = np.stack([logm(jac[i] + iden) for i in range(bs)])
    trace_jac = np.diagonal(log_jac, axis1=1, axis2=2).sum(1)
    return trace_jac


def power_series_full_jac_exact_trace(Fx, x, k):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation with full
    jacobian and exact trace
    
    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :return: Tr(Ln(I + df/dx))
    """
    _, jac = compute_log_det(x, Fx)
    jacPower = jac
    summand = torch.zeros_like(jacPower)
    for i in range(1, k+1):
        if i > 1:
            jacPower = torch.matmul(jacPower, jac)
        if (i + 1) % 2 == 0:
            summand += jacPower / (np.float(i))
        else: 
            summand -= jacPower / (np.float(i)) 
    trace = torch.diagonal(summand, dim1=1, dim2=2).sum(1)
    return trace


def power_series_matrix_logarithm_trace(Fx, x, k, n):
    """
    Fast-boi Tr(Ln(d(Fx)/dx)) using power-series approximation
    biased but fast
    :param Fx: output of f(x)
    :param x: input
    :param k: number of power-series terms  to use
    :param n: number of Hitchinson's estimator samples
    :return: Tr(Ln(I + df/dx))
    """
    # trace estimation including power series
    outSum = Fx.sum(dim=0)
    dim = list(outSum.shape)
    dim.insert(0, n)
    dim.insert(0, x.size(0))
    u = torch.randn(dim).to(x.device)
    trLn = 0
    for j in range(1, k + 1):
        if j == 1:
            vectors = u
        # compute vector-jacobian product
        vectors = [torch.autograd.grad(Fx, x, grad_outputs=vectors[:, i],
                                       retain_graph=True, create_graph=True)[0] for i in range(n)]
        # compute summand
        vectors = torch.stack(vectors, dim=1)
        vjp4D = vectors.view(x.size(0), n, 1, -1)
        u4D = u.view(x.size(0), n, -1, 1)
        summand = torch.matmul(vjp4D, u4D)
        # add summand to power series
        if (j + 1) % 2 == 0:
            trLn += summand / np.float(j)
        else:
            trLn -= summand / np.float(j)
    trace = trLn.mean(dim=1).squeeze()
    return trace


def compute_log_det(inputs, outputs):
    log_det_fn = log_det_2x2 if inputs.size()[1] == 2 else log_det_other
    batch_size = outputs.size(0)
    outVector = torch.sum(outputs,0).view(-1)
    outdim = outVector.size()[0]
    jac = torch.stack([torch.autograd.grad(outVector[i], inputs,
                                     retain_graph=True, create_graph=True)[0].view(batch_size, outdim) for i in range(outdim)], dim=1)
    log_det = torch.stack([log_det_fn(jac[i,:,:]) for i in range(batch_size)], dim=0)
    return log_det, jac


def log_det_2x2(x):
    return torch.log(x[0,0]*x[1,1]-x[0,1]*x[1,0])


def log_det_other(x):
    return torch.logdet(x)


def weak_bound(sigma, d, n_terms):
    """
    Returns a bound on |Tr(Ln(A)) - PowerSeries(A, n_terms)|
    :param sigma: lipschitz constant of block
    :param d: dimension of data
    :param n_terms: number of terms in our estimate
    :return:
    """
    inf_sum = -np.log(1. - sigma)
    fin_sum = 0.
    for k in range(1, n_terms + 1):
        fin_sum += (sigma**k) / k

    return d * (inf_sum - fin_sum)


if __name__ == "__main__":
    print() 
