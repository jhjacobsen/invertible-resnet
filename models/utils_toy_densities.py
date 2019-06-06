import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from models.model_utils import get_all_params
from torch.autograd.gradcheck import zero_gradients
from models.density_modeling_utils import compute_distributions
import os
import sys
import math
import numpy as np
import pdb
import timeit
from joblib import Parallel, delayed
from models.toy_data import inf_train_gen
from matrix_utils import compute_log_det
import multiprocessing
VISDOMWINDOWS = {}

criterion = nn.CrossEntropyLoss()



def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def line_plot(viz, title, x, y):
    if title in VISDOMWINDOWS:
        window = VISDOMWINDOWS[title]
        viz.line(X=[x], Y=[y], win=window, update='append')
    else:
        window = viz.line(X=[x], Y=[y], opts={'title': title})
        VISDOMWINDOWS[title] = window


def scatter_plot(viz, title, x):
    if x.size()[1] > 2:
      x = x[:,:2]
    if title in VISDOMWINDOWS:
        window = VISDOMWINDOWS[title]
        viz.scatter(X=x, win=window, update='replace')
    else:
        window = viz.scatter(X=x, opts={'title': title})
        VISDOMWINDOWS[title] = window


def unwrap_stack(vars):
    vars = [v.contiguous().view(-1) for v in vars if v is not None]
    vars = torch.cat(vars)
    return vars


def bits_per_dim(logpx, inputs):
    return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) 


def train(model, dataset, dim, its, batch_size, lr, use_cuda,
          weight_decay=1e-5, nesterov=False,
          coeff=0.9, svd_clipping=True, viz=None,
          brute_force=True, brute_force_logging=False):

    model.train()
    # create optimzer
    params = lambda: filter(lambda p: p.requires_grad, model.parameters())  # FUCK pytorch
    if nesterov:
        optimizer = optim.SGD(params(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=nesterov)
    else:
        optimizer = optim.Adam(params(), lr=lr, weight_decay=weight_decay)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])

    print('|  Number of Trainable Parameters: {}'.format(num_params))
    print('\n=> Training {}, LR = {}'.format(dataset, lr))

    for i in range(its):
        # set to train mode for spectral normalization
        model.train()
        # data batch
        inputs = inf_train_gen(dataset, batch_size=batch_size, dim=dim).float()
        # maybe move to gpu
        if use_cuda:
            inputs = inputs.cuda()

        optimizer.zero_grad()
        inputs = Variable(inputs, requires_grad=True)

        # Forward Propagation
        out_bij, p_z_g_y, trace, fj_trace = model(inputs)

        if brute_force or brute_force_logging:
            # brute-force logdet and jacobian
            log_det, jac = compute_log_det(inputs, out_bij)
            line_plot(viz, "log|df/dz|", i, log_det.mean().item())
            line_plot(viz, "|trace - log|df/dz|", i, (log_det - trace).abs().mean().item())
            line_plot(viz, "trace - log|df/dz|", i, (log_det - trace).mean().item())
            log_det_grad = torch.autograd.grad(log_det.mean(), params(), allow_unused=True, retain_graph=True)
            log_det_grad = unwrap_stack(log_det_grad)
            trace_grad = torch.autograd.grad(trace.mean(), params(), allow_unused=True, retain_graph=True)
            trace_grad = unwrap_stack(trace_grad)
            #line_plot(viz, "||Grad(log|df/dz|) - Grad(trace)||^2", i, ((log_det_grad - trace_grad) ** 2).sum().item())
            #line_plot(viz, "Grad(log|df/dz|) - Grad(trace)", i, (log_det_grad - trace_grad).mean().item())
            # logging of sigma from spectral norm
            sigmas = []
            for k in model.state_dict().keys():
                if 'bottleneck' and 'weight_orig' in k:                    
                    sigma = model.state_dict()[k[:-5] + '_sigma']
                    sigmas.append(sigma.item())
            sigmas = np.array(sigmas)
            line_plot(viz, "sigma all layers", i, sigmas)
            # logging of bias
            line_plot(viz, "bias power series: log det - fj_trace", i, (log_det - fj_trace).mean().item())
            # logging of bias in gradients
            fj_trace_grad = torch.autograd.grad(fj_trace.mean(), params(), allow_unused=True, retain_graph=True)
            fj_trace_grad = unwrap_stack(log_det_grad)
            line_plot(viz, "bias of gradients (max)", i, torch.max(log_det - fj_trace).item())
        
        # decide which loss for training
        if brute_force:
            p_x_g_y = p_z_g_y + log_det
        else:
            p_x_g_y = p_z_g_y + trace
            if brute_force_logging:
                line_plot(viz, "bits/dim brute force", i, bits_per_dim( (p_z_g_y + log_det).mean().item(),inputs))

        loss = -(p_x_g_y).mean()
        line_plot(viz, "bits/dim", i, bits_per_dim(-loss.item(),inputs))
        line_plot(viz, "logp(x)", i, -loss.item())
        line_plot(viz, "logp(z)", i, p_z_g_y.mean().item())
        line_plot(viz, "trace", i, trace.mean().item())

        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update
        
        # do clippling if selected
        if i % 20 == 0:
            if svd_clipping:
                clip_fc_layer(model, coeff, use_cuda)

        if i % 10 == 0:
            sys.stdout.write('\r')
            if brute_force or brute_force_logging:
                sys.stdout.write(
                    '| Iter [{}/{}]\t\t logp(x): {:.3f} Log|df/dx|: {:.3f} Trace: {:.3f}'.format(
                        i + 1, its,
                        -loss.item(), log_det.mean().item(), trace.mean().item()))
            else:
                sys.stdout.write(
                    '| Iter [{}/{}]\t\t logp(x): {:.3f} Trace: {:.3f}'.format(
                        i + 1, its,
                        -loss.item(), trace.mean().item()))
            sys.stdout.flush()

        #if i == (its - 1):
        if i % 100 == 0:
            # set to eval mode for spectral normalization
            model.eval()
            out_bij, p_z_g_y, _, _ = model(inputs)
            samples = model.module.sample(batch_size)
            recon = model.module.inverse(model(inputs)[0])
            line_plot(viz, "recons_loss", i, ((inputs - recon)**2).sum(dim=1).mean().cpu().item())
            scatter_plot(viz, "samples", samples.cpu())
            scatter_plot(viz, "data", inputs.cpu())
            scatter_plot(viz, "recons", recon.cpu())
            scatter_plot(viz, "latents", out_bij.cpu())


def test(model, dataset, dim, use_cuda, fname, its=100, batch_size=100, brute_force=True):
    model.eval()
    name_sample = "samples_{}".format(dataset)
    name_zs = "zs_{}".format(dataset)
    loss = 0.
    for i in range(its):
        inputs = inf_train_gen(dataset, batch_size=batch_size, dim=dim).float()
        # maybe move to gpu
        if use_cuda:
            inputs = inputs.cuda()

        inputs = Variable(inputs, requires_grad=True)

        # Forward Propagation
        out_bij, p_z_g_y, trace = model(inputs)
        # approximate logdet
        tmp_trace = torch.zeros_like(trace[0])
        for k in range(len(trace)):
            tmp_trace += trace[k]
        trace = tmp_trace

        if brute_force:
            # brute-force logdet and jacobian
            log_det, jac = compute_log_det(inputs, out_bij)
            p_x_g_y = p_z_g_y + log_det
        else:
            p_x_g_y = p_z_g_y + trace

        loss += -(p_x_g_y).mean()

    loss /= its
    samples = model.module.sample(batch_size)
    np.save(name_sample, samples.cpu().data.numpy())
    np.save(name_zs, out_bij.cpu().data.numpy())
    print("\n| Testing \t\t\t logp(x) = {:.4f}".format(-loss), flush=True)
    print('| Saving model...', flush=True)
    state = {'model': model, 'loss': loss}
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/'+dataset+os.sep
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    torch.save(state, save_point+fname+'.t7')
    return loss


def _clipping_comp_fc(param, key, coeff, use_cuda):
    if "bottleneck" not in key or "weight" not in key:  # non fc-parameters
        return
    # compute SVD
    fcMatrix = param.data
    U, D, V = torch.svd(fcMatrix, some=True)
    if torch.max(D) > coeff: 
        # first projection onto given norm ball
        Dclip = torch.min(D, torch.tensor(coeff).cuda())
        matrixClipped = torch.matmul(U * Dclip, torch.t(V))
        param.data = matrixClipped
    return


def clip_fc_layer(model, coeff, use_cuda):
    for param, key in zip(model.parameters(), model.module.state_dict().keys()):
        _clipping_comp_fc(param, key, coeff, use_cuda)


def test_sum():
    #test: sum up to one
    clip_fc_layer(model, coeff, use_cuda)
    minx = -10
    maxx = 10
    num = 10000
    inputs = np.array([np.linspace(minx, maxx, num),]).transpose()
    inputs = torch.tensor(inputs).float().cuda()
    inputs = Variable(inputs, requires_grad=True)
    out_bij, p_z_g_y, trace, gt_trace = model(inputs)
    log_det = compute_log_det(model, inputs, out_bij)
    log_det = torch.tensor(log_det).cuda()
    p_x_g_y = p_z_g_y + log_det.view(log_det.size(0), 1)
    p_x = ((p_x_g_y) / float(np.log(2.) * np.prod(inputs.shape[1:])))
    exp_p_x = torch.exp(p_x_g_y)
    sum_p = exp_p_x.sum(dim=0)* ((maxx - minx) / num)
    pdb.set_trace()

    #test: sum up to one in 2D
    minx = -4
    maxx = 4
    num = 100
    x = np.linspace(minx, maxx, num)
    xx, yy = np.meshgrid(x, x)
    xx = xx.reshape(num*num)
    yy = yy.reshape(num*num)
    inputs = np.vstack([xx, yy]).transpose()
    inputs = torch.tensor(inputs).float().cuda()
    inputs = Variable(inputs, requires_grad=True)
    out_bij, p_z_g_y, trace, gt_trace = model(inputs)
    log_det = compute_log_det(model, inputs, out_bij)
    log_det = torch.tensor(log_det).cuda()
    p_x_g_y = p_z_g_y + log_det#.view(log_det.size(0), 1)
    p_x = p_x_g_y
    exp_p_x = torch.exp(p_x_g_y)
    sum_p = exp_p_x.sum(dim=0) * ((maxx - minx)**2 / num**2)
    pdb.set_trace()

