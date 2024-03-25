import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import pdb






def gradient_update(f_i, y_i, z_i, gamma):
    output = OrderedDict()
    t1 = f_i.state_dict()
    for name in f_i.state_dict():
        if ('num_batches_tracked' in name or 'running_var' in name or 'running_mean' in name):
            del t1[name]
    # pdb.set_trace()
    grad_dict = {k: v.grad for k, v in zip(t1, f_i.parameters())}
    # for v in f_i.parameters():
    #     print(v.shape)
    # for k in t1:
    #     print(k)
    #     print(f_i.state_dict()[k].shape)

    # pdb.set_trace()
    black_list = ['num_batches_tracked', 'running_var', 'running_mean']
    for name in f_i.state_dict():
        # try:
        if not ('num_batches_tracked' in name or 'running_var' in name or 'running_mean' in name):
            # print(name)
            # if 'linear' in name:
            output[name] = f_i.state_dict()[name] - gamma*(y_i[name] + z_i[name] + grad_dict[name])
            #     # pdb.set_trace()
            # else:
            # output[name] = f_i.state_dict()[name] - gamma*(grad_dict[name])
        else:
            output[name] = f_i.state_dict()[name]
        # except:
        #     pdb.set_trace()
    # pdb.set_trace()
    return output


def update_znew(f, x_old, y, W, N_in, i, nc_agents, T, gamma, z_new_i):
    cluster = i//nc_agents
    ic = i % nc_agents

    z_r = OrderedDict()
    for name in f[i].state_dict():
        z_r[name] = W[cluster][ic, ic]*(f[i].state_dict()[name] - x_old[i][name] + gamma*y[i][name])
        for j in N_in[cluster][ic]:
            z_r[name] += W[cluster][ic, j]*(f[cluster*nc_agents + j].state_dict()[name] - x_old[cluster*nc_agents + j][name] \
                                            + gamma*y[cluster*nc_agents + j][name])
        # pdb.set_trace()
        z_new_i[name] += 1/(T*gamma)*(f[i].state_dict()[name] - x_old[i][name] + gamma*y[i][name] - z_r[name])
    
    return z_new_i


def update_xnew(f, R, N_in, i, nc_agents, x_new_i):
    cluster = i//nc_agents
    ic = i % nc_agents
    for name in f[i].state_dict():
        x_new_i[name] = R[cluster][ic, ic]*f[i].state_dict()[name]
        for j in N_in[cluster][ic]:
            x_new_i[name] += R[cluster][ic, j]*f[cluster*nc_agents + j].state_dict()[name]

    return x_new_i