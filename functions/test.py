import torch
import numpy as np
from .gradfn import func
from .utils import get_mse_fstar, build_asy_list
import random
import scipy.io as sio
import pdb


def Test1(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
    N = input.shape[0]
    dim = input.shape[1]
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    label_list = list(torch.split(labels.squeeze(), n, dim=0))
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)

    f = []
    for (inp, lab) in zip(input_list, label_list):
        f_i = func(inp, lab, lam=lam, opt=opt)
        f.append(f_i)

    x_list = [x.type(torch.double) for i in range(n_agents)]
    x_new = [x.type(torch.double) for i in range(n_agents)]
    sigma = [x.type(torch.double) for i in range(n_agents)]

    # y = [0 for i in range(n_agents)]
    y = [0 for i in range(n_agents)]
    y_new = [f[i].g(x_list[i]) for i in range(n_agents)]


    com_count = 0
    gamma = step_c*1

    error_list = []

    error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1))/n_agents).item())

    com_count = 0
    com_iter = 0
    fg_old = torch.zeros(n_agents, dim).type(torch.double).to(x.device)
    iters = 0
    print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    # while (com_count < opt.c_rounds):
    nc_agents = n_agents//opt.n_cluster

    J = torch.ones(nc_agents, nc_agents)/nc_agents
    rho_m = max(abs(torch.linalg.eig(R[0] - J)[0]))  # The largest eigenvalue
    # pdb.set_trace()
    # rho_m = 1 - (nc_agents - opt.sample_num)/nc_agents
    # W = torch.block_diag(*torch.stack(R))
    # Big_J = torch.ones(n_agents, n_agents)/n_agents

    T = int(opt.p_inv)
    for r in range(int(opt.c_rounds)):
        f_sum = [0 for i in range(n_agents)]

        for t in range(T):
            for i in range(n_agents):
                f_sum[i] += f[i].g(x_list[i])/T
                x_list[i] = x_list[i] - gamma*(f[i].g(x_list[i]) + y[i])

                # x_list[i] = x_new[i]

        # com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        # com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        # com_list = [item for sublist in com_list for item in sublist]

        x_cen = 0
        for n_c in range(n_agents):
            x_cen += 1/n_agents*(x_list[n_c])

        '''broadcast'''


        total_f = torch.mean(torch.stack(f_sum), dim= 0)
        fg = 0
        for i in range(n_agents):
            fg += 1/n_agents*f[i].g(x_cen)
        for i in range(n_agents):

            y[i] = total_f - f_sum[i]
            
            x_list[i] = x_cen

            # pdb.set_trace()

        # for n_c in com_list:
        #     cluster = n_c//nc_agents
        #     y[n_c] = avg_y[cluster]

        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(x_cen.reshape(dim, 1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % opt.p_iter == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list


def Test2(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
    N = input.shape[0]
    dim = input.shape[1]
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    label_list = list(torch.split(labels.squeeze(), n, dim=0))
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)

    AAt = input.transpose(0, 1) @ input
    t0 = torch.real(torch.linalg.eig(AAt)[0])
    kappa = torch.max(t0)/torch.min(t0)
    L = torch.max(t0)

    f = []
    for (inp, lab) in zip(input_list, label_list):
        f_i = func(inp, lab, lam=lam, opt=opt)
        f.append(f_i)

    x_list = [x.type(torch.double) for i in range(n_agents)]
    x_new = [x.type(torch.double) for i in range(n_agents)]
    sigma = [x.type(torch.double) for i in range(n_agents)]

    # y = [0 for i in range(n_agents)]
    y = [0 for i in range(n_agents)]
    y_new = [0 for i in range(n_agents)]

    com_count = 0
    gamma = step_c*1

    error_list = []

    error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1))/n_agents).item())

    com_count = 0
    com_iter = 0
    fg_old = torch.zeros(n_agents, dim).type(torch.double).to(x.device)
    iters = 0
    print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    # while (com_count < opt.c_rounds):
    nc_agents = n_agents//opt.n_cluster

    J = torch.ones(nc_agents, nc_agents)/nc_agents
    rho_m = max(abs(torch.linalg.eig(R[0] - J)[0]))  # The largest eigenvalue
    # pdb.set_trace()
    # rho_m = 1 - (nc_agents - opt.sample_num)/nc_agents
    # W = torch.block_diag(*torch.stack(R))
    # Big_J = torch.ones(n_agents, n_agents)/n_agents
    beta = [1 for i in range(n_agents)]
    fg_history = [[] for i in range(n_agents)]
    current_fg = [[] for i in range(n_agents)]

    T = int(opt.p_inv)
    for r in range(int(opt.c_rounds)):
        # f_sum = [0 for i in range(n_agents)]
        y_sum = [0 for i in range(n_agents)]

        for t in range(T):
            for i in range(n_agents):
                x_new = x_list[i] - gamma*(f[i].g(x_list[i]) + y[i])
                y_sum[i] += y[i]

                current_fg[i].append(f[i].g(x_list[i]))
                if len(fg_history[i]) != 0:
                    y[i] = y[i] + (1/n_agents*f[i].g(x_list[i]) - f[i].g(x_list[i])) - (1/n_agents*fg_history[i][t] - fg_history[i][t])
                x_list[i] = x_new

        # com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        # com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        # com_list = [item for sublist in com_list for item in sublist]

        x_cen = 0
        for n_c in range(n_agents):
            x_cen += 1/n_agents*(x_list[n_c] + gamma*y_sum[n_c])

        '''broadcast'''

        for i in range(n_agents):
        
            # pdb.set_trace()
            y[i] = y[i] + 1/(T*gamma)*(x_list[i] - x_cen)
            x_list[i] = x_cen

        fg_history = current_fg
        current_fg = [[] for i in range(n_agents)]

            # pdb.set_trace()

        # for n_c in com_list:
        #     cluster = n_c//nc_agents
        #     y[n_c] = avg_y[cluster]

        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(x_cen.reshape(dim, 1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % opt.p_iter == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list
