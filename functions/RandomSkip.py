import torch
import numpy as np
from .gradfn import func
from .utils import get_mse_fstar, build_asy_list
import random
import scipy.io as sio
import pdb


def RandomSkip(x, input, labels, skip_list, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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


    z_list = [f[i].g(x_list[i]) for i in range(n_agents)]
    z_new = [f[i].g(x_list[i]) for i in range(n_agents)]
    diff = torch.zeros(n_agents, dim).type(torch.double).to(x.device)

    rho = torch.zeros(n_agents, dim).type(torch.double).to(x.device)
    for i in range(n_agents):
        rho[i] = z_list[i]

    rho_tilde = torch.zeros(n_agents, n_agents, dim).type(torch.double).to(x.device)

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

    for t in range(int(opt.c_rounds)):

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 0:
                fg_old[i] = f[i].g(x_list[i])
                x_list[i] = x_list[i] - gamma*(z_list[i] + diff[i])
                fg_new = f[i].g(x_list[i])
                diff[i] = diff[i] + fg_new - fg_old[i]

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 1:
                fg_old[i] = f[i].g(x_list[i])
                sigma[i] = x_list[i] - gamma*(z_list[i] + diff[i])

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 1:
                x_new[i] = R[i, i]*sigma[i]
                for j in N_in[i]:
                    x_new[i] += R[i, j]*sigma[j]

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 1:
                fg_new = f[i].g(x_new[i])
                z_new[i] = C[i, i]*z_list[i] + (diff[i] + fg_new - fg_old[i])
                for j in N_in[i]:
                    z_new[i] += C[i, j]*(rho[j] - rho_tilde[i, j])
                    rho_tilde[i, j] = rho[j]
                    # z_new[i] += C[i, j]*z_list[j]
                diff[i] = 0

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 1:
                rho[i] += z_new[i]
                z_list[i] = z_new[i]
                x_list[i] = x_new[i]



        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % 1000 == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list


def Skip2(x, input, labels, skip_list, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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

    z_list = [f[i].g(x_list[i]) for i in range(n_agents)]
    z_new = [f[i].g(x_list[i]) for i in range(n_agents)]
    diff = torch.zeros(n_agents, dim).type(torch.double).to(x.device)


    rho = torch.zeros(n_agents, dim).type(torch.double).to(x.device)
    for i in range(n_agents):
        rho[i] = z_list[i]

    rho_tilde = torch.zeros(n_agents, n_agents, dim).type(torch.double).to(x.device)

    com_count = 0
    gamma = step_c*1

    error_list = []

    error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1))/n_agents).item())

    com_count = 0
    com_iter = 0
    fg_old = torch.zeros(n_agents, dim).type(torch.double).to(x.device)
    iters = 0
    print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    for t in range(int(opt.c_rounds)):

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 0:
                fg_old[i] = f[i].g(x_list[i])
                x_list[i] = x_list[i] - gamma*(z_list[i] + diff[i])
                fg_new = f[i].g(x_list[i])
                diff[i] = diff[i] + fg_new - fg_old[i]

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 1:
                fg_old[i] = f[i].g(x_list[i])
                sigma[i] = x_list[i] - gamma*(z_list[i] + diff[i])

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 1:
                x_new[i] = R[i, i]*sigma[i]
                for j in N_in[i]:
                    x_new[i] += R[i, j]*sigma[j]

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 1:
                fg_new = f[i].g(x_new[i])
                z_new[i] = C[i, i]*z_list[i] + (diff[i] + fg_new - fg_old[i])
                for j in N_in[i]:
                    z_new[i] += C[i, j]*(rho[j])
                    # rho_tilde[i, j] = rho[j]
                    # z_new[i] += C[i, j]*z_list[j]
                diff[i] = 0

        for i in range(n_agents):
            theta_it = skip_list[i][t]
            if theta_it == 1:
                rho[i] = z_new[i]
                z_list[i] = z_new[i]
                x_list[i] = x_new[i]


        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % 1000 == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list
