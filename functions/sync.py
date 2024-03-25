import torch
import numpy as np
from .gradfn import func
from .utils import get_mse_fstar, build_asy_list
import random
import scipy.io as sio
import pdb


def NEW_SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, fstar, c_rounds, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
    phi = [z_list[i] for i in range(n_agents)]
    fg_mean_old = [z_list[i] for i in range(n_agents)]
    fg_mean_new = [0 for i in range(n_agents)]
    bound_list = []

    x_nu_old = torch.mean(torch.stack(x_list), dim=0)

    while (com_count < c_rounds):

        # round_len = random.randint(n_agents, round_max)

        # i = round_list[iters]
        # i = com_count % n_agents
        # i = 0
        fg_old = [0 for i in range(n_agents)]
        fg_new = [0 for i in range(n_agents)]
        z_i_half = [0 for i in range(n_agents)]
        # z_start = [0 for i in range(n_agents)]
        x_new = [0 for i in range(n_agents)]
        diff = [0 for i in range(n_agents)]

        # for i in range(n_agents):
        #     z_start[i] = z_list[i]

        for t in range(int(T)):
            for i in range(n_agents):
                '''gradient descent'''
                v[i] = x_list[i] - gamma*(phi[i] + diff[i])
                fg_old[i] = f[i].g(x_list[i])
                fg_mean_new[i] += fg_old[i]
            # fg_newnew = f[i].g(v[i])
            # v[i] = v[i] - gamma*(z_list[i] + fg_newnew - f[i].g(x_list[i]))
            '''consensus on x'''
            for i in range(n_agents):
                if t == T-1:
                    fg_mean_new[i] = (1/T)*fg_mean_new[i]
                    x_new[i] = R[i, i]*v[i]
                    for j in N_in[i]:
                        x_new[i] += R[i, j]*v[j]
                else:
                    x_new[i] = v[i].clone()

            '''update z and push rho'''
            for i in range(n_agents):
                fg_new[i] = f[i].g(x_new[i])
                x_list[i] = x_new[i]
                diff[i] = diff[i] + (fg_new[i] - fg_old[i])
                # pdb.set_trace()
                if t == T-1:
                    z_i_half[i] = C[i, i]*(z_list[i] + (fg_mean_new[i] - fg_mean_old[i]))
                    for j in N_in[i]:
                        z_i_half[i] += C[i, j]*(z_list[j] + (fg_mean_new[j] - fg_mean_old[j]))
                        # rho_tilde[i, j] = rho[j]

                    # z_i_half[i] += (fg_mean_new[i] - fg_mean_old[i])


            if t == T-1:


                for i in range(n_agents):
                    # rho[i] += z_i_half[i]
                    # z_list[i] = C[i, i]*z_i_half[i]
                    z_list[i] = z_i_half[i]
                    phi[i] = z_list[i] + (fg_new[i] - fg_mean_new[i])
                    fg_mean_old[i] = fg_mean_new[i]
                    fg_mean_new[i] = 0

        # pdb.set_trace()
        # if com_count == n_agents:
        #     pdb.set_trace()

        '''debug'''
        x_nu_new = torch.mean(torch.stack(x_list), dim=0)

        cond1 = -2*torch.matmul(x_nu_new - x_nu_old, x_nu_old - x_opt) - gamma*torch.norm(x_nu_new - x_nu_old) > 0

        gamma_bound = -2*torch.matmul(x_nu_new - x_nu_old, x_nu_old - x_opt)/torch.norm(x_nu_new - x_nu_old)

        print("The condition holds: ", cond1,"   ", gamma_bound.item())

        bound_list.append(gamma_bound.item())
        # pdb.set_trace()

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

    return error_list, bound_list


def SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, fstar, c_rounds, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
        # z_start = [0 for i in range(n_agents)]
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
                if t == T-1:
                    x_new[i] = R[i, i]*v[i]
                    for j in N_in[i]:
                        x_new[i] += R[i, j]*v[j]
                else:
                    x_new[i] = v[i].clone()

            '''update z and push rho'''
            for i in range(n_agents):
                fg_new[i] = f[i].g(x_new[i])
                x_list[i] = x_new[i]
                # pdb.set_trace()
                if t == T-1:
                    z_i_half[i] = C[i, i]*z_list[i]
                    # z_i_half[i] = z_list[i] + (fg_new[i] - fg_old[i])

                    for j in N_in[i]:
                        # t1 = z_list[j]
                        # t2 = rho[j] - rho_tilde[i, j]
                        z_i_half[i] += C[i, j]*z_list[j]
                        # z_i_half[i] += C[i, j]*(rho[j] - rho_tilde[i, j])
                        # if com_count != 0:
                        #     pdb.set_trace()
                        rho_tilde[i, j] = rho[j]
                    # diff = z_list[i] - z_start[i]
                    z_i_half[i] += (fg_new[i] - fg_old[i]) + diff[i]

                else:
                    # z_i_half[i] = z_list[i] + (fg_new[i] - fg_old[i])
                    # z_list[i] = z_i_half[i]
                    diff[i] = diff[i] + (fg_new[i] - fg_old[i])

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


def OLD_SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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

    while (com_count < opt.c_rounds):

        # round_len = random.randint(n_agents, round_max)

        # i = round_list[iters]
        # i = com_count % n_agents
        # i = 0
        fg_old = [0 for i in range(n_agents)]
        fg_new = [0 for i in range(n_agents)]
        z_i_half = [0 for i in range(n_agents)]
        # z_start = [0 for i in range(n_agents)]
        x_new = [0 for i in range(n_agents)]
        # for i in range(n_agents):
        #     z_start[i] = z_list[i]

        for t in range(int(T)):
            for i in range(n_agents):
                '''gradient descent'''
                v[i] = x_list[i] - gamma*(z_list[i])
                fg_old[i] = f[i].g(x_list[i])
            # fg_newnew = f[i].g(v[i])
            # v[i] = v[i] - gamma*(z_list[i] + fg_newnew - f[i].g(x_list[i]))
            '''consensus on x'''
            for i in range(n_agents):
                if t == T-1:
                    x_new[i] = R[i, i]*v[i]
                    for j in N_in[i]:
                        x_new[i] += R[i, j]*v[j]
                else:
                    x_new[i] = v[i].clone()

            '''update z and push rho'''
            for i in range(n_agents):
                fg_new[i] = f[i].g(x_new[i])
                x_list[i] = x_new[i]
                # pdb.set_trace()
                if t == T-1:
                    # z_i_half[i] = C[i, i]*z_list[i]
                    z_i_half[i] = z_list[i] + (fg_new[i] - fg_old[i])

                    for j in N_in[i]:
                        # z_i_half[i] += C[i, j]*z_list[j]
                        z_i_half[i] += C[i, j]*(rho[j] - rho_tilde[i, j])
                        rho_tilde[i, j] = rho[j]
                    # diff = z_list[i] - z_start[i]
                    # z_i_half[i] += (fg_new[i] - fg_old[i])

                else:
                    z_i_half[i] = z_list[i] + (fg_new[i] - fg_old[i])
                    z_list[i] = z_i_half[i]

            if t == T-1:
                for i in range(n_agents):
                    rho[i] += z_i_half[i]
                    z_list[i] = C[i,i]*z_i_half[i]
                    # z_list[i] = z_i_half[i]

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
