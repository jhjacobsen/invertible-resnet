import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.autograd import Variable
from torch._six import inf
from models.viz_utils import line_plot, scatter_plot, images_plot
import os
import sys
import math
import numpy as np
import json
import pdb

import time



from joblib import Parallel, delayed
import multiprocessing
criterion = nn.CrossEntropyLoss()

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def learning_rate(init, epoch):
    optim_factor = 0
    if epoch > 160:
        optim_factor = 3
    elif epoch > 120:
        optim_factor = 2
    elif epoch > 60:
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def out_im(im):
    imc = torch.clamp(im, -.5, .5)
    return imc + .5


def bits_per_dim(logpx, inputs):
    return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)


def train(args, model, optimizer, epoch, trainloader, trainset, viz, use_cuda, train_log):
    model.train()
    correct = 0
    total = 0

    # update lr for this epoch (for classification only)
    if not args.densityEstimation:
        lr = learning_rate(args.lr, epoch)
        update_lr(optimizer, lr)
    else:
        lr = args.lr

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('|  Number of Trainable Parameters: ' + str(params))
    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, lr))
          
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        cur_iter = (epoch - 1) * len(trainloader) + batch_idx
        # if first epoch use warmup
        if epoch - 1 <= args.warmup_epochs:
            this_lr = args.lr * float(cur_iter) / (args.warmup_epochs * len(trainloader))
            update_lr(optimizer, this_lr)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        

        if args.densityEstimation: # density estimation
            _, logpz, trace = model(inputs)  # Forward Propagation
            # compute loss
            logpx = logpz + trace
            loss = bits_per_dim(logpx, inputs).mean()
        else: # classification
            out, _ = model(inputs)
            loss = criterion(out, targets) # Loss
        
        # logging for sigmas. NOTE: needs to be done before backward-call
        if args.densityEstimation and args.log_verbose:
            if batch_idx % args.log_every == 0:
                sigmas = []
                for k in model.state_dict().keys():
                    if 'bottleneck' and 'weight_orig' in k:               
                        sigma = model.state_dict()[k[:-5] + '_sigma']
                        sigmas.append(sigma.item())
                sigmas = np.array(sigmas)
                line_plot(viz, "sigma all layers", cur_iter, sigmas)
                
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update
                
        if args.densityEstimation: # logging for density estimation
            if batch_idx % args.log_every == 0:
                mean_trace = trace.mean().item()
                mean_logpz = logpz.mean().item()
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t%% bits/dim: %.3f Trace: %.3f  logp(z) %.3f'
                                 % (epoch, args.epochs, batch_idx+1,
                                    (len(trainset)//args.batch)+1, loss,  mean_trace, mean_logpz))
                sys.stdout.flush()
                line_plot(viz, "bits/dim", cur_iter, loss.item())
                line_plot(viz, "logp(z)", cur_iter, mean_logpz)
                line_plot(viz, "log|df/dz|", cur_iter, mean_trace)
                # file logging
                log_dict = {"iter": cur_iter, "loss": loss.item(), "logpz": mean_logpz, "logdet": mean_trace, "epoch": epoch}
                train_log.write("{}\n".format(json.dumps(log_dict)))
                train_log.flush()

                if args.log_verbose:
                    # grad_norm_2 = sum((p.grad.norm()**2).item() for p in model.parameters() if p.grad is not None)
                    grad_norm_inf = max(p.grad.data.abs().max().item() for p in model.parameters() if p.grad is not None)
                    # line_plot(viz, "grad_norm_2", cur_iter, grad_norm_2)
                    line_plot(viz, "grad_norm_inf", cur_iter, grad_norm_inf)
                    # log actnorm scaling
                    if not args.noActnorm:
                        actnorm_scales = []
                        actnorm_scales_min = []
                        actnorm_l2 = []
                        for k in model.state_dict().keys():
                            if 'actnorm' and '_log_scale' in k:
                                scale = torch.max(model.state_dict()[k])
                                scale_min = torch.min(model.state_dict()[k])
                                l2 = torch.norm(model.state_dict()[k])
                                actnorm_scales.append(scale.item())
                                actnorm_scales_min.append(scale_min.item())
                                actnorm_l2.append(l2.item())
                        actnorm_scales = np.array(actnorm_scales)
                        actnorm_scales_min = np.array(actnorm_scales_min)
                        actnorm_l2 = np.array(actnorm_l2)
                        line_plot(viz, "max actnorm scale per layer", cur_iter, actnorm_scales)
                        line_plot(viz, "min actnorm scale per layer", cur_iter, actnorm_scales_min)
                        line_plot(viz, "l2 norm of actnorm scale per layer", cur_iter, actnorm_l2)  
                    # learned prior logging
                    if not args.fixedPrior:
                        prior_scales_max = torch.max(model.state_dict()['module.prior_logstd'])
                        prior_scales_min = torch.min(model.state_dict()['module.prior_logstd'])
                        line_plot(viz, "max prior scale", cur_iter, prior_scales_max.item())
                        line_plot(viz, "min prior scale", cur_iter, prior_scales_min.item())  

        else: # logging for classification
            _, predicted = torch.max(out.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()    
            if batch_idx % 1 == 0:
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f'
                                 % (epoch, args.epochs, batch_idx+1,
                                    (len(trainset)//args.batch)+1, loss.data.item(),
                                    100.*correct.type(torch.FloatTensor)/float(total)))
                sys.stdout.flush()


def test(best_result, args, model, epoch, testloader, viz, use_cuda, test_log):
    model.eval()
    objective = 0.
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)

        if args.densityEstimation:
            z, logpz, trace = model(inputs)
            logpx = logpz + trace
            loss = bits_per_dim(logpx, inputs)

            objective += -loss.cpu().sum().item()

            # visualization and samples
            if batch_idx == 0:
                x_re = model.module.inverse(z, 10) if use_cuda else model.inverse(z, 10)
                err = (inputs - x_re).abs().sum()
                line_plot(viz, "recons err", epoch, err.item())
                bs = inputs.size(0)
                samples = model.module.sample(bs, 10) if use_cuda else model.sample(bs, 10)
                im_dir = os.path.join(args.save_dir, 'ims')
                try_make_dir(im_dir)
                torchvision.utils.save_image(samples.cpu(),
                                             os.path.join(im_dir, "samples_{}.jpg".format(epoch)),
                                             int(bs**.5), normalize=True)
                torchvision.utils.save_image(inputs.cpu(),
                                             os.path.join(im_dir, "data_{}.jpg".format(epoch)),
                                             int(bs ** .5), normalize=True)
                torchvision.utils.save_image(x_re.cpu(),
                                             os.path.join(im_dir, "recons_{}.jpg".format(epoch)),
                                             int(bs ** .5), normalize=True)
                images_plot(viz, "data", out_im(inputs).cpu())
                images_plot(viz, "recons", out_im(x_re).cpu())
                images_plot(viz, "samples", out_im(samples).cpu())
                del x_re, err, samples

            del z, logpz, trace, logpx, loss

        else:
            out, out_bij = model(inputs)
            _, predicted = torch.max(out.data, 1)
            objective += predicted.eq(targets.data).sum().item()
            del out, out_bij, _, predicted

        total += targets.size(0)
        del inputs, targets

    objective = float(objective) / float(total)
    line_plot(viz, "test bits/dim" if args.densityEstimation else "test acc", epoch, objective)
    print("\n| Validation Epoch #%d\t\t\tobjective =  %.4f" % (epoch, objective), flush=True)
    if objective > best_result:
        print('\n| Saving Best model...\t\t\tobjective = %.4f%%' % (objective), flush=True)
        state = {
            'model': model if use_cuda else model,
            'objective': objective,
            'epoch': epoch,
        }

        try_make_dir(args.save_dir)
        torch.save(state, os.path.join(args.save_dir, 'checkpoint.t7'))
        best_result = objective
    else:
        print('\n| Not best... {:.4f} < {:.4f}'.format(objective, best_result), flush=True)

    # log to file
    test_log.write("{} {}\n".format(epoch, objective))
    test_log.flush()
    return best_result


def _determine_shapes(model):
    in_shapes = model.module.get_in_shapes()
    i = 0
    j = 0
    shape_list = list()
    for key, _ in model.named_parameters():
        if "bottleneck" not in key or "weight" not in key:  # non conv-parameters
            shape_list.append(None)
            continue
        shape_list.append(tuple(in_shapes[j]))
        if i == 2:
            i = 0
            j+= 1
        else:
            i+=1
    return shape_list


def _clipping_comp(param, key, coeff, input_shape, use_cuda):
    if "bottleneck" not in key or "weight" not in key:  # non conv-parameters
        return
    # compute SVD via FFT
    convKernel = param.data.cpu().numpy()
    input_shape = input_shape[1:]
    fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
    t_fft_coeff = np.transpose(fft_coeff)
    U, D, V = np.linalg.svd(t_fft_coeff, compute_uv=True, full_matrices=False)
    if np.max(D) > coeff: 
        # first projection onto given norm ball
        Dclip = np.minimum(D, coeff)
        coeffsClipped = np.matmul(U * Dclip[..., None, :], V)
        convKernelClippedfull = np.fft.ifft2(coeffsClipped, axes=[0, 1]).real
        # 1) second projection back to kxk filter kernels
        # and transpose to undo previous transpose operation (used for batch SVD)
        kernelSize1, kernelSize2 = convKernel.shape[2:]
        convKernelClipped = np.transpose(convKernelClippedfull[:kernelSize1, :kernelSize2])          
        # reset kernel (using in-place update)
        if use_cuda:
            param.data += torch.tensor(convKernelClipped).float().cuda() - param.data
        else:
            param.data += torch.tensor(convKernelClipped).float() - param.data
    return


def clip_conv_layer(model, coeff, use_cuda):
    shape_list = _determine_shapes(model)
    num_cores = multiprocessing.cpu_count()
    for (key, param), shape in zip(model.named_parameters(), shape_list):
        _clipping_comp(param, key, coeff, shape, use_cuda)
    return

def interpolate(model, testloader, testset, epoch, use_cuda, best_acc, dataset, fname):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        out, out_bij, _ = model(inputs)
        loss = criterion(out, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(out.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model

    acc = 100.*correct.type(torch.FloatTensor)/float(total)
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.4f%%" %(epoch, loss.data[0], acc),flush=True)

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.4f%%' % (acc), flush=True)
        state = {
                'model': model if use_cuda else model,
                'acc': acc,
                'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+dataset+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+fname+'.t7')
        best_acc = acc
    return best_acc

softmax = nn.Softmax(dim=1)
