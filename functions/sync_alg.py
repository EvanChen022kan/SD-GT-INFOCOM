import torch
import numpy as np
from .gradfn import func
from .utils import get_mse_fstar, build_asy_list
import random
import scipy.io as sio
import pdb


def SYNC_ALG_1(x, input, labels, R, C, N_in, N_out, x_opt, fstar, c_rounds,  lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
    # pdb.set_trace()
    # N_out = N_in # for undirected graph
    f = []
    for (inp, lab) in zip(input_list, label_list):
        f_i = func(inp, lab, lam=lam, opt=opt)
        f.append(f_i)

    x_list = [x.type(torch.double) for i in range(n_agents)]
    com_count = 0
    gamma = step_c*1
    error_list = []

    error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1))/n_agents).item())

    com_count = 0
    com_iter = 0
    iters = 0
    print("(SYNC ALG1)[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
    # xt = x

    z_list = [f[i].g(x_list[i]) for i in range(n_agents)]
    z_hat = torch.mean(torch.stack(z_list, dim=0), dim=0)
    c = [-z_list[i] + z_hat for i in range(n_agents)]


    while (com_count < c_rounds):

        x_K = [x_list[i].clone() for i in range(n_agents)]
        for t in range(int(T)):
            for i in range(n_agents):
                '''gradient descent'''
                fg = f[i].g(x_K[i])
                x_K[i] = x_K[i] - gamma*(fg + c[i])

        for i in range(n_agents):
            z_list[i] = (1/(T*gamma))*(x_list[i] - x_K[i])

        for i in range(n_agents):
            c[i] = c[i] - z_list[i] + C[i,i]*z_list[i]
            for j in N_in[i]:
                c[i] += C[i, j]*z_list[j]
        
        xi_new = [0 for i in range(n_agents)]
        for i in range(n_agents):
            xi_new[i] = R[i, i]*(x_list[i] - T*gamma*z_list[i])
            for j in N_in[i]:
                xi_new[i] += R[i, j]*(x_list[j] - T*gamma*z_list[j])

        x_list = xi_new

        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        if iters % int(1/4*opt.c_rounds) == 0:
            print("(SYNC ALG1)[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')


    return error_list


def SYNC_ALG_2(x, input, labels, R, C, N_in, N_out, x_opt, fstar,c_rounds, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
    # pdb.set_trace()
    # N_out = N_in # for undirected graph
    f = []
    for (inp, lab) in zip(input_list, label_list):
        f_i = func(inp, lab, lam=lam, opt=opt)
        f.append(f_i)

    x_list = [x.type(torch.double) for i in range(n_agents)]
    com_count = 0
    gamma = step_c*1
    error_list = []

    error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1))/n_agents).item())

    com_count = 0
    com_iter = 0
    iters = 0
    print("(SYNC ALG2)[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
    # xt = x

    z_list = [f[i].g(x_list[i]) for i in range(n_agents)]
    z_hat = torch.mean(torch.stack(z_list, dim=0), dim=0)
    y = [0 for i in range(n_agents)]

    # C = R = (1/n_agents)*torch.ones(n_agents, n_agents).type(torch.double).to(x.device)
    # N_in = [[] for i in range(n_agents)]
    # N_out = [[] for i in range(n_agents)]
    # for i in range(n_agents):
    #     for j in range(n_agents):
    #         if R[i, j] != 0 and i != j:
    #             N_in[i].append(j)
    #         if C[j, i] != 0 and i != j:
    #             N_out[i].append(j)

    while (com_count < c_rounds):
        # pdb.set_trace()
        phi = [x_list[i].clone() for i in range(n_agents)]
        for t in range(int(T)):
            for i in range(n_agents):
                '''gradient descent'''
                fg = f[i].g(phi[i])
                phi[i] = phi[i] - gamma*(fg + y[i])

        # pdb.set_trace()

        for i in range(n_agents):
            x_list[i] = R[i, i]*phi[i]
            # grad = C[i, i]*phi[i]
            for j in N_in[i]:
                x_list[i] += R[i,j]*phi[j]
                # grad += C[i, j]*phi[j]

            y[i] = y[i] + 1/(T*gamma)*(phi[i] - x_list[i])

        iters += 1
        com_iter += 1
        com_count += 1
        # if x_list[0].dtype != torch.float64:
        #     pdb.set_trace()
        error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        if iters % int(1/4*opt.c_rounds) == 0:
            print("(SYNC ALG2)[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    return error_list


def SYNC_ALG_3(x, input, labels, R, C, N_in, N_out, x_opt, fstar, c_rounds, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
    # pdb.set_trace()
    # N_out = N_in # for undirected graph
    f = []
    for (inp, lab) in zip(input_list, label_list):
        f_i = func(inp, lab, lam=lam, opt=opt)
        f.append(f_i)

    x_list = [x.type(torch.double) for i in range(n_agents)]
    # mat = sio.loadmat('x_init.mat')['X']
    # mat = torch.from_numpy(mat).type(torch.double).to(x.device)
    # for i in range(n_agents):
    #     x_list[i] = mat[:, i]

    z_list = [f[i].g(x_list[i]) for i in range(n_agents)]

    rho = torch.zeros(n_agents, dim).type(torch.double).to(x.device)
    v = torch.zeros(n_agents, dim).type(torch.double).to(x.device)
    rho_tilde = torch.zeros(n_agents, n_agents, dim).type(torch.double).to(x.device)

    # def global_f(x): return torch.mean(torch.stack([f[i].f(x) for i in range(n_agents)]))
    # def global_g(x): return torch.mean(torch.stack([f[i].g(x) for i in range(n_agents)], dim=0), dim=0)
    def global_f(x): return torch.sum(torch.stack([f[i].f(x) for i in range(n_agents)]))
    def global_g(x): return torch.sum(torch.stack([f[i].g(x) for i in range(n_agents)], dim=0), dim=0)
    com_count = 0
    # gamma = 3/L
    gamma = step_c*1
    # pdb.set_trace()

    round_max = 30
    # round_len = round_max
    round_list = [i for i in range(n_agents)] + [random.randint(0, n_agents-1) for i in range(round_max - n_agents)]
    round_list = build_asy_list(opt.c_rounds, round_max, n_agents)

    mat = sio.loadmat('act.mat')['act_list']
    round_list = (mat-1).squeeze()
    # x_opt = x.clone()
    error_list = []
    # error_list.append((global_f(x) - fstar).item()) # save the initial gap
    error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1))/n_agents).item())
    # t1 = torch.stack(x_list, dim=1)
    # t2 = torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1)
    # pdb.set_trace()
    com_count = 0
    com_iter = 0
    iters = 0
    print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
    central_flag = opt.test_cen
    # xt = x
    # for i in range(50000):
    #     xt = xt - 1/(L*n_agents)*global_g(xt)
    #     if i % 1000 == 0:

    #         print(torch.norm(xt-x_opt).item())

    while (com_count < c_rounds):

        # round_len = random.randint(n_agents, round_max)

        # i = round_list[iters]
        # i = com_count % n_agents
        # i = 0
        fg_old = [0 for i in range(n_agents)]
        fg_new = [0 for i in range(n_agents)]
        z_i_half = [0 for i in range(n_agents)]
        x_new = [0 for i in range(n_agents)]
        diff = [0 for i in range(n_agents)]

        # for i in range(n_agents):
        #     z_start[i] = z_list[i]

        for t in range(int(T)):
            for i in range(n_agents):
                '''gradient descent'''
                v[i] = x_list[i] - gamma*(z_list[i] + diff[i])
                fg_old[i] = f[i].g(x_list[i])
            # fg_newnew = f[i].g(v[i])
            # v[i] = v[i] - gamma*(z_list[i] + fg_newnew - f[i].g(x_list[i]))
            '''consensus on x'''
            for i in range(n_agents):
                x_new[i] = v[i].clone()

            '''update z and push rho'''
            for i in range(n_agents):
                fg_new[i] = f[i].g(x_new[i])
                diff[i] = diff[i] + (fg_new[i] - fg_old[i])

                x_list[i] = x_new[i]
                # pdb.set_trace()
                if t == T-1:
                    x_list[i] = R[i, i]*v[i]
                    for j in N_in[i]:
                        x_list[i] += R[i, j]*v[j]

                    z_i_half[i] = R[i, i]*(z_list[i] + diff[i])
                    # z_i_half[i] = z_list[i] + (fg_new[i] - fg_old[i])

                    for j in N_in[i]:
                        # t1 = z_list[j]
                        # t2 = rho[j] - rho_tilde[i, j]
                        z_i_half[i] += R[i, j]*(z_list[j] + diff[j])
                        # z_i_half[i] += C[i, j]*(rho[j] - rho_tilde[i, j])
                        # if com_count != 0:
                        #     pdb.set_trace()
                        # rho_tilde[i, j] = rho[j]
                    # diff = z_list[i] - z_start[i]
                    # z_i_half[i] += (fg_new[i] - fg_old[i]) + diff[i]


            if t == T-1:
                for i in range(n_agents):
                    rho[i] += z_i_half[i]
                    # z_list[i] = C[i, i]*z_i_half[i]
                    z_list[i] = z_i_half[i]

        # pdb.set_trace()
        # if com_count == n_agents:
        #     pdb.set_trace()

        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % int(1/4*opt.c_rounds) == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

        # if iters % T == 0:
        #     # x_hat = torch.mean(torch.stack(x_list), dim=0)

        #     com_count += 1
        #     com_iter = 0
        #     # error_list.append((global_f(x_hat) - fstar).item())

        #     if com_count % int(0.1*opt.c_rounds) == 0:
        #         print("[%d agents]Communication Count: " % n_agents, com_count, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    return error_list
