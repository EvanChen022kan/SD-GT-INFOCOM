import torch
import numpy as np
from .gradfn import func
from .utils import get_mse_fstar, build_asy_list
import random
import scipy.io as sio
import pdb


def NEW_ASYSONATA(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
    for i in range(n_agents):
        rho[i] = z_list[i]
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

    # xt = x
    # for i in range(50000):
    #     xt = xt - 1/(L*n_agents)*global_g(xt)
    #     if i % 1000 == 0:

    #         print(torch.norm(xt-x_opt).item())
    fg_mean_old = [z_list[i] for i in range(n_agents)]
    phi = [z_list[i] for i in range(n_agents)]
    # fg_mean_new = [0 for i in range(n_agents)]

    while (com_count < opt.c_rounds):

        # round_len = random.randint(n_agents, round_max)

        i = round_list[iters]
        diff = 0
        fg_mean_new = 0
        # i = com_count % n_agents
        # i = 0
        for t in range(int(T)):
            '''gradient descent'''
            v[i] = x_list[i] - gamma*(phi[i] + diff)
            fg_old = f[i].g(x_list[i])
            fg_mean_new += fg_old
            # fg_newnew = f[i].g(v[i])
            # v[i] = v[i] - gamma*(z_list[i] + fg_newnew - f[i].g(x_list[i]))
            '''consensus on x'''
            # if t == T-1:
            #     x_new = R[i, i]*v[i]
            #     for j in N_in[i]:
            #         x_new += R[i, j]*v[j]
            if t == T-1:
                fg_mean_new = (1/T)*fg_mean_new
                x_new = R[i, i]*v[i]
                for j in N_in[i]:
                    x_new += R[i, j]*v[j]
            else:
                x_new = v[i].clone()

            '''update z and push rho'''
            fg_new = f[i].g(x_new)
            x_list[i] = x_new
            # pdb.set_trace()
            diff += (fg_new - fg_old)

            if t == T-1:
                z_i_half = C[i, i]*z_list[i]
                for j in N_in[i]:
                    z_i_half += C[i, j]*(rho[j] - rho_tilde[i, j])
                    rho_tilde[i, j] = rho[j]
                # z_i_half += (fg_new - fg_old) + diff

                z_i_half += (fg_mean_new - fg_mean_old[i])

                z_list[i] = z_i_half
                rho[i] += z_i_half

                phi[i] = z_list[i] + (fg_new - fg_mean_new)
                fg_mean_old[i] = fg_mean_new

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

def ASYSONATA(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
    for i in range(n_agents):
        rho[i] = z_list[i]
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

    # xt = x
    # for i in range(50000):
    #     xt = xt - 1/(L*n_agents)*global_g(xt)
    #     if i % 1000 == 0:

    #         print(torch.norm(xt-x_opt).item())

    while (com_count < opt.c_rounds):

        # round_len = random.randint(n_agents, round_max)

        i = round_list[iters]
        diff = 0
        # i = com_count % n_agents
        # i = 0
        for t in range(int(T)):
            '''gradient descent'''
            v[i] = x_list[i] - gamma*(z_list[i] + diff)
            fg_old = f[i].g(x_list[i])
            # fg_newnew = f[i].g(v[i])
            # v[i] = v[i] - gamma*(z_list[i] + fg_newnew - f[i].g(x_list[i]))
            '''consensus on x'''
            # if t == T-1:
            #     x_new = R[i, i]*v[i]
            #     for j in N_in[i]:
            #         x_new += R[i, j]*v[j]
            if t == T-1:
                x_new = R[i, i]*v[i]
                for j in N_in[i]:
                    x_new += R[i, j]*v[j]
            else:
                x_new = v[i].clone()

            '''update z and push rho'''
            fg_new = f[i].g(x_new)
            x_list[i] = x_new
            # pdb.set_trace()
            if t == T-1:
                z_i_half = C[i, i]*z_list[i]
                for j in N_in[i]:
                    z_i_half += C[i, j]*(rho[j] - rho_tilde[i, j])
                    rho_tilde[i, j] = rho[j]
                z_i_half += (fg_new - fg_old) + diff
                
                z_list[i] = z_i_half
                rho[i] += z_i_half
            else:
                # z_list[i] = z_i_half
                diff += (fg_new - fg_old)
                # rho[i] += z_i_half


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


def ASY_GT(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
    for i in range(n_agents):
        rho[i] = z_list[i]
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

    # xt = x
    # for i in range(50000):
    #     xt = xt - 1/(L*n_agents)*global_g(xt)
    #     if i % 1000 == 0:

    #         print(torch.norm(xt-x_opt).item())

    while (com_count < opt.c_rounds):

        # round_len = random.randint(n_agents, round_max)

        i = round_list[iters]
        diff = 0
        # i = com_count % n_agents
        # i = 0
        for t in range(int(T)):
            '''gradient descent'''
            v[i] = x_list[i] - gamma*(z_list[i] + diff)
            fg_old = f[i].g(x_list[i])
            # fg_newnew = f[i].g(v[i])
            # v[i] = v[i] - gamma*(z_list[i] + fg_newnew - f[i].g(x_list[i]))
            '''consensus on x'''
            # if t == T-1:
            #     x_new = R[i, i]*v[i]
            #     for j in N_in[i]:
            #         x_new += R[i, j]*v[j]
            if t == T-1:
                x_new = R[i, i]*v[i]
                for j in N_in[i]:
                    x_new += R[i, j]*v[j]
            else:
                x_new = v[i].clone()

            '''update z and push rho'''
            fg_new = f[i].g(x_new)
            x_list[i] = x_new
            # pdb.set_trace()
            if t == T-1:
                z_i_half = C[i, i]*z_list[i]
                for j in N_in[i]:
                    z_i_half += C[i, j]*(rho[j])
                z_i_half += (fg_new - fg_old) + diff

                z_list[i] = z_i_half
                rho[i] = z_i_half
            else:
                # z_list[i] = z_i_half
                diff += (fg_new - fg_old)
                # rho[i] += z_i_half

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


def ASY_SGD(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
    for i in range(n_agents):
        rho[i] = z_list[i]
    v = torch.zeros(n_agents, dim).type(torch.double).to(x.device)
    rho_tilde = torch.zeros(n_agents, n_agents, dim).type(torch.double).to(x.device)

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

    com_count = 0
    com_iter = 0
    iters = 0
    print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    while (com_count < opt.c_rounds):
        i = round_list[iters]
        diff = 0

        for t in range(int(T)):
            '''gradient descent'''
            v[i] = x_list[i] - gamma*(z_list[i] + diff)
            fg_old = f[i].g(x_list[i])
            if t == T-1:
                x_new = R[i, i]*v[i]
                for j in N_in[i]:
                    x_new += R[i, j]*v[j]
            else:
                x_new = v[i].clone()

            '''update z and push rho'''
            fg_new = f[i].g(x_new)
            x_list[i] = x_new
            if t == T-1:
                z_list[i] = (fg_new - fg_old) + diff
            else:
                diff += (fg_new - fg_old)

        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        if iters % 1000 == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list

def OLD_ASYSONATA(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c = 1,  opt=None):
    N = input.shape[0]
    dim = input.shape[1]
    n = N//n_agents
    input_list = list(torch.split(input, n, dim = 0))
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
        f_i = func(inp, lab, lam = lam, opt=opt)
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

    # xt = x
    # for i in range(50000):
    #     xt = xt - 1/(L*n_agents)*global_g(xt)
    #     if i % 1000 == 0:
            
    #         print(torch.norm(xt-x_opt).item())


    
    while (com_count < opt.c_rounds):
        
        # round_len = random.randint(n_agents, round_max)
        
        i = round_list[iters]
        # i = com_count % n_agents
        # i = 0
        for t in range(int(T)):
            '''gradient descent'''
            v[i] = x_list[i] - gamma*(z_list[i])
            fg_old = f[i].g(x_list[i])
            # fg_newnew = f[i].g(v[i])
            # v[i] = v[i] - gamma*(z_list[i] + fg_newnew - f[i].g(x_list[i]))
            '''consensus on x'''
            # if t == T-1:
            #     x_new = R[i, i]*v[i]
            #     for j in N_in[i]:
            #         x_new += R[i, j]*v[j]
            if t == T-1:
                x_new = R[i, i]*v[i]
                for j in N_in[i]:
                    x_new += R[i, j]*v[j]
            else:
                x_new = v[i].clone()

            '''update z and push rho'''
            fg_new = f[i].g(x_new)
            z_i_half = z_list[i] + (fg_new - fg_old)
            x_list[i] = x_new
            # pdb.set_trace()
            if t == T-1:
                for j in N_in[i]:
                    z_i_half += C[i, j]*(rho[j] - rho_tilde[i, j])
                    rho_tilde[i, j] = rho[j]
                z_list[i] = C[i,i]*z_i_half
                rho[i] += z_i_half
            else:
                z_list[i] = z_i_half
                # rho[i] += z_i_half

            
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

        if iters % 1000 == 0:
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


