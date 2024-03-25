import torch
import numpy as np
from .gradfn import func
from .utils import get_mse_fstar, build_asy_list
import random
import scipy.io as sio
import pdb

def SemiFL_ScatterGT2(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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

    y = [0 for i in range(n_agents)]

    # z = [0 for i in range(n_agents)]
    z = [f[i].g(x_list[i]) for i in range(n_agents)]
    z_new = [0 for i in range(n_agents)]

    
    rho = [0 for i in range(n_agents)]
    rho_tilde = [0 for i in range(n_agents)]
    fg_old = [0 for i in range(n_agents)]

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
    T = int(opt.p_inv)
    for r in range(int(opt.c_rounds)):
        # z_total = [0 for i in range(n_agents)]
        rho = [0 for i in range(n_agents)]

        for t in range(T):
            for i in range(n_agents):
                fg_old[i] = f[i].g(x_list[i])
                # x_new[i] = x_list[i] - gamma*(f[i].g(x_list[i]) + z[i] + y[i])
                x_new[i] = x_list[i] - gamma*(z[i] + y[i])

                # rho[i] += f[i].g(x_list[i])
                # z_total[i] += z[i]

            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents
                # print("agent %d: " % i)
                x_list[i] = R[cluster][ic, ic]*x_new[i]
                z_new[i] = R[cluster][ic, ic]*z[i]
                # pdb.set_trace()

                for j in N_in[cluster][ic]:
                    x_list[i] += R[cluster][ic, j]*x_new[cluster*nc_agents + j]
                    z_new[i] += R[cluster][ic, j]*z[cluster*nc_agents + j]
                # z[i] = z[i] + 1/(gamma)*(x_new[i] - x_list[i])
            for i in range(n_agents):
                z[i] = z_new[i] + f[i].g(x_list[i]) - fg_old[i]

        com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]


        '''aggregate'''
        x_cen = 0
        rho_cen = 0
        for n_c in com_list:
            x_cen += 1/opt.n_cluster*(x_list[n_c])
            # rho_cen += 1/opt.n_cluster*(rho[n_c])

        '''broadcast'''
        # clus = 0
        for n_c in com_list:
            '''correction term update'''
            y[n_c] = y[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen)
            # y[n_c] = (1/T)*(rho_cen - rho[n_c])

            x_list[n_c] = x_cen

        '''scatter'''
        for n_c in com_list:
            cluster = n_c//nc_agents
            for k in range(nc_agents):
                y[cluster*nc_agents + k] = y[n_c]

        # pdb.set_trace()
        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % 500 == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list

def SemiFL_ScatterGT(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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

    y = [0 for i in range(n_agents)]

    z = [0 for i in range(n_agents)]
    # z_new = [0 for i in range(n_agents)]

    rho = [0 for i in range(n_agents)]
    rho_tilde = [0 for i in range(n_agents)]

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
    T = int(opt.p_inv)
    for r in range(int(opt.c_rounds)):

        for t in range(T):
            for i in range(n_agents):
                x_new[i] = x_list[i] - gamma*(f[i].g(x_list[i]) + y[i])

            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents
                # print("agent %d: " % i)
                x_list[i] = R[cluster][ic, ic]*x_new[i]
                # pdb.set_trace()

                for j in N_in[cluster][ic]:
                    x_list[i] += R[cluster][ic, j]*x_new[cluster*nc_agents + j]
                # z[i] = z[i] + 1/gamma*(x_new[i] - x_list[i])

        com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]


        '''aggregate'''
        x_cen = 0
        for n_c in com_list:
            x_cen += 1/opt.n_cluster*x_list[n_c]

        '''broadcast'''
        # clus = 0
        for n_c in com_list:
            '''correction term update'''
            y[n_c] = y[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen)
            x_list[n_c] = x_cen

        '''scatter'''
        for n_c in com_list:
            cluster = n_c//nc_agents
            for k in range(nc_agents):
                y[cluster*nc_agents + k] = y[n_c]

        # pdb.set_trace()
        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
        error_list.append((torch.norm(x_cen.reshape(dim, 1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % 500 == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list

def SemiFL_GT(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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


    y = [0 for i in range(n_agents)]

    z = [0 for i in range(n_agents)]
    z_new = [0 for i in range(n_agents)]

    rho = [0 for i in range(n_agents)]
    rho_tilde = [0 for i in range(n_agents)]


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
    rho_m = (nc_agents - opt.sample_num)/nc_agents
    T = int(opt.p_inv)
    for r in range(int(opt.c_rounds)):
        
        for t in range(T):
            for i in range(n_agents):
                x_new[i] = x_list[i] - gamma*(f[i].g(x_list[i]) + y[i])
                # z_new[i] = z[i] + f[i].g(x_new[i]) - f[i].g(x_list[i])

            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents
                x_list[i] = R[cluster][ic, ic]*x_new[i]
                for j in N_in[cluster][ic]:
                    x_list[i] += R[cluster][ic, j]*x_new[cluster*nc_agents + j]
                # x_list[i] = x_new[i]



        # com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        com_list = [item for sublist in com_list for item in sublist]


        for i in range(n_agents):
            rho[i] += z[i]

        '''aggregate'''
        x_cen = 0
        z_cen = 0
        for n_c in com_list:
            x_cen += 1/len(com_list)*(x_list[n_c] + T*gamma*y[n_c])


        '''broadcast'''
        # clus = 0
        for n_c in com_list:
            '''correction term update'''
            y[n_c] = y[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen)
            # y[n_c] = rho_m*y[n_c] + (1 - rho_m)*(y[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen))

            x_list[n_c] = x_cen
        # pdb.set_trace()

        '''scatter'''
        # if opt.scatter:
        #     for n_c in com_list:
        #         cluster = n_c//nc_agents
        #         for k in range(nc_agents):
        #             y[cluster*nc_agents + k] = y[n_c]

        # pdb.set_trace()
        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
        error_list.append((torch.norm(x_cen.reshape(dim,1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % opt.p_iter == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list


def SemiFL(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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

    y = [0 for i in range(n_agents)]

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
    T = int(opt.p_inv)
    for r in range(int(opt.c_rounds)):

        for t in range(T):
            for i in range(n_agents):
                x_new[i] = x_list[i] - gamma*(f[i].g(x_list[i]))

            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents

                x_list[i] = R[cluster][ic, ic]*x_new[i]
                for j in N_in[cluster][ic]:
                    x_list[i] += R[cluster][ic, j]*x_new[cluster*nc_agents + j]

        # com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        com_list = [item for sublist in com_list for item in sublist]

        '''aggregate'''
        x_cen = 0
        for n_c in com_list:
            x_cen += 1/len(com_list)*x_list[n_c]
        '''broadcast'''
        for n_c in com_list:
            '''correction term update'''
            # y[n_c] = 1/(T*gamma)*(x_list[n_c] - x_cen)
            x_list[n_c] = x_cen
            # pdb.set_trace()

        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()

        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
        error_list.append((torch.norm(x_cen.reshape(dim, 1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        if iters % opt.p_iter == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list


def SemiFL_GT2_old(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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

    z = [0 for i in range(n_agents)]
    z_new = [0 for i in range(n_agents)]

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
    T = int(opt.p_inv)
    for r in range(int(opt.c_rounds)):

        for t in range(T):
            for i in range(n_agents):
                # fg_old[i] = f[i].g(x_list[i])
                z[i] = z_new[i]
                x_new[i] = x_list[i] - gamma*(f[i].g(x_list[i]) + z[i] + y[i])
                # x_new[i] = x_list[i] - gamma*y[i]

            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents
                # print("agent %d: " % i)
                x_list[i] = R[cluster][ic, ic]*(x_new[i])
                for j in N_in[cluster][ic]:
                    x_list[i] += R[cluster][ic, j]*(x_new[cluster*nc_agents + j])

                # z_new[i] = z[i] + y[i] + 1/(gamma)*(x_new[i] - x_list[i])
        
        com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        # com_list = [i for i in range(n_agents)]

        x_cen = 0
        y_cen = 0
        for n_c in com_list:
            # x_cen += 1/opt.n_cluster*(x_list[n_c] + T*gamma*(y[n_c] + z[n_c]))
            x_cen += 1/opt.n_cluster*(x_list[n_c] + T*gamma*(y[n_c] + z[n_c]))

        '''broadcast'''
        t_x_cen = 0
        for i in range(n_agents):
            t_x_cen += 1/n_agents*(x_list[i] + T*gamma*y[i])

        for i in range(n_agents):
            x_new[i] = R[cluster][ic, ic]*(x_list[i] + T*gamma*(z[i]))
            for j in N_in[cluster][ic]:
                x_new[i] += R[cluster][ic, j]*(x_list[cluster*nc_agents + j] + T*gamma*(z[cluster*nc_agents + j]))
            z_new[i] = z[i] + 1/(T*gamma)*(x_list[i] - x_new[i])

        for n_c in com_list:
            # y[n_c] = y[n_c] + z[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen) - z_new[n_c]
            y[n_c] = y[n_c] + z[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen) - z_new[n_c]
            # y[n_c] = y[n_c] + 1/(T*gamma)*(x_list[n_c] - t_x_cen)
            # fg_old[n_c] = f[n_c].g(x_list[n_c])
            x_list[n_c] = x_cen
            # y[n_c] = y_cen
            # y[n_c] += f[n_c].g(x_list[n_c]) - fg_old[n_c]


        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % opt.p_iter == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list


def SemiFL_GT2(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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

    z = [f[i].g(x_list[i]) for i in range(n_agents)]
    z_new = [0 for i in range(n_agents)]

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
    rho_m = (nc_agents - opt.sample_num)/nc_agents

    T = int(opt.p_inv)
    for r in range(int(opt.c_rounds)):

        for t in range(T):
            for i in range(n_agents):
                x_new[i] = x_list[i] - gamma*(f[i].g(x_list[i]) + y[i])

            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents
                # print("agent %d: " % i)
                x_list[i] = R[cluster][ic, ic]*(x_new[i])
                for j in N_in[cluster][ic]:
                    x_list[i] += R[cluster][ic, j]*(x_new[cluster*nc_agents + j])



        # com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        com_list = [item for sublist in com_list for item in sublist]

        x_cen = 0
        for n_c in com_list:
            x_cen += 1/len(com_list)*(x_list[n_c] + T*gamma*(y[n_c]))

        '''broadcast'''
        # t_x_cen = 0
        # for i in range(n_agents):
        #     t_x_cen += 1/n_agents*(x_list[i] + T*gamma*y[i])

        # for i in range(n_agents):
        #     x_new[i] = R[cluster][ic, ic]*(x_list[i] + T*gamma*(z[i]))
        #     for j in N_in[cluster][ic]:
        #         x_new[i] += R[cluster][ic, j]*(x_list[cluster*nc_agents + j] + T*gamma*(z[cluster*nc_agents + j]))
        #     z_new[i] = z[i] + 1/(T*gamma)*(x_list[i] - x_new[i])

        for n_c in com_list:
            y[n_c] = y[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen)
            # y[n_c] = rho_m*y[n_c] + (1 - rho_m)*(y[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen))
            # pdb.set_trace()
            x_list[n_c] = x_cen
        

        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % opt.p_iter == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list


def SemiFL_GT3(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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
    # y_new = [0 for i in range(n_agents)]

    # y_tmp = [torch.zeros(opt.dim).to(x.device) for i in range(n_agents)]

    z = [f[i].g(x_list[i]) for i in range(n_agents)]
    z_new = [0 for i in range(n_agents)]

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
    # rho_m *= 3
    # pdb.set_trace()
    # rho_m = 1 - (nc_agents - opt.sample_num)/nc_agents
    # W = torch.block_diag(*torch.stack(R))
    # Big_J = torch.ones(n_agents, n_agents)/n_agents

    T = int(opt.p_inv)
    for r in range(int(opt.c_rounds)):
        # y_tmp = [0 for i in range(n_agents)]
        y_tmp = torch.zeros((opt.dim, n_agents, T), dtype = torch.double).to(x.device)

        # pdb.set_trace()
        for t in range(T):
            for i in range(n_agents):
                x_new[i] = x_list[i] - gamma*(f[i].g(x_list[i]) + y[i])

            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents
                # print("agent %d: " % i)
                x_list[i] = R[cluster][ic, ic]*(x_new[i])
                for j in N_in[cluster][ic]:
                    x_list[i] += R[cluster][ic, j]*(x_new[cluster*nc_agents + j])

                if t == 0:
                    y_tmp[:, i, t] += R[cluster][ic, ic]*(y[i])
                    for j in N_in[cluster][ic]:
                        y_tmp[:, i, t] += R[cluster][ic, j]*(y[cluster*nc_agents + j])
                else:
                    y_tmp[:, i, t] += R[cluster][ic, ic]*(y_tmp[:, i, t-1])
                    for j in N_in[cluster][ic]:
                        y_tmp[:, i, t] += R[cluster][ic, j]*(y_tmp[:, cluster*nc_agents + j, t-1])


                # x_list[i] = x_new[i]

        # pdb.set_trace()

        # com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        com_list = [item for sublist in com_list for item in sublist]

        x_cen = 0
        for n_c in com_list:
            x_cen += 1/len(com_list)*(x_list[n_c] + T*gamma*(y[n_c]))

        '''broadcast'''
        # t_x_cen = 0
        # for i in range(n_agents):
        #     t_x_cen += 1/n_agents*(x_list[i] + T*gamma*y[i])

        # for i in range(n_agents):
        #     x_new[i] = R[cluster][ic, ic]*(x_list[i] + T*gamma*(z[i]))
        #     for j in N_in[cluster][ic]:
        #         x_new[i] += R[cluster][ic, j]*(x_list[cluster*nc_agents + j] + T*gamma*(z[cluster*nc_agents + j]))
        #     z_new[i] = z[i] + 1/(T*gamma)*(x_list[i] - x_new[i])
        G_W = torch.zeros(n_agents, n_agents)
        for i in range(n_agents):
            for j in range(n_agents):
                if i==j:
                    G_W[i, j] = 1
                if i in com_list and j in com_list:
                    G_W[i,j] = 1/len(com_list)
        
        avg_y = [0 for i in range(opt.n_cluster)]

        # pdb.set_trace()
        y_tmp_sum = torch.mean(y_tmp, dim = 2)
        # y_tmp_sum = T*y_tmp[:, :, 0]

        # pdb.set_trace()
        # for i in range(n_agents):
        #     y[i] = (opt.sample_num/nc_agents)*y_tmp_sum[:, i]

        for n_c in com_list:
            # y[n_c] = y[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen)
            # y[n_c] = rho_m*y[n_c] + (1 - rho_m)*(y[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen))
            beta = opt.beta
            y[n_c] = y_tmp_sum[:, n_c] + beta*(1/(T*gamma)*(x_list[n_c] - x_cen))
            # y[n_c] += beta*(1/(T*gamma)*(x_list[n_c] - x_cen))


            # y[n_c] = beta*(y[n_c] + 1/(T*gamma)*(x_list[n_c] - x_cen))

            x_list[n_c] = x_cen

            cluster = n_c//nc_agents
            avg_y[cluster] += y[n_c]/opt.sample_num

            # pdb.set_trace()
        
        # for n_c in com_list:
        #     cluster = n_c//nc_agents
        #     y[n_c] = avg_y[cluster]


        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(x_cen.reshape(dim,1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % opt.p_iter == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return error_list


def SemiFL_GT4(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, sample_num_opt = 1, n_agents=10, L=10, step_c=1, control=0, energy_list=None, opt=None):
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

    # x = torch.zeros(dim).to(x.device)
    x_list = [x.type(torch.double) for i in range(n_agents)]
    # x_list2 = [x.type(torch.double) for i in range(n_agents)]
    x_new = [x.type(torch.double) for i in range(n_agents)]
    x_old = [x.type(torch.double) for i in range(n_agents)]

    sigma = [x.type(torch.double) for i in range(n_agents)]

    test = 0

    # y = [0 for i in range(n_agents)]
    y = [torch.zeros(dim).type(torch.double).to(x.device) for i in range(n_agents)]
    yt = [0 for i in range(n_agents)]

    z = [0 for i in range(n_agents)]



    com_count = 0
    gamma = torch.tensor(step_c*1, dtype=torch.double)

    error_list = []

    energy_cost_list = []

    energy_count = 0

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
    # rho_m *= 3
    # pdb.set_trace()
    # rho_m = 1 - (nc_agents - opt.sample_num)/nc_agents
    # W = torch.block_diag(*torch.stack(R))
    # Big_J = torch.ones(n_agents, n_agents)/n_agents

    # T = int(opt.p_inv)
    z = [0 for i in range(n_agents)]
    z_new = [0 for i in range(n_agents)]

    rho_m = torch.tensor(1 - opt.sample_num/nc_agents, dtype=torch.double)
    x_cen = 0
    y_cen = 0
    c = [0 for i in range(n_agents)]

    y_tmp = torch.zeros((dim, n_agents, T+1), dtype=torch.double).to(x.device)
    z_tmp = torch.zeros((dim, n_agents, T+1), dtype=torch.double).to(x.device)

    sample_count = [1 for i in range(n_agents)]

    com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
    com_list = [item for sublist in com_list for item in sublist]
    error_z = []

    y_old = [0 for i in range(n_agents)]
    cluster_y = [0 for i in range(n_agents//nc_agents)]

    x_cen = 0
    # energy_list = [random.randint(1, 200) for i in range(opt.n_cluster)]

    for r in range(int(opt.c_rounds)):
        # y_tmp = [0 for i in range(n_agents)]
        grad_f = torch.zeros((n_agents, T, opt.dim), dtype=torch.double).to(x.device)
        gf_mix = torch.zeros((opt.dim, n_agents, T+1), dtype=torch.double).to(x.device)
        x_mix = torch.zeros((opt.dim, n_agents, T+1), dtype=torch.double).to(x.device)
        for i in range(n_agents):
            x_mix[:, i, 0] = x_list[i]
            z_new[i] = z[i]
        x_init = torch.stack(x_list)

        for t in range(T):
            for i in range(n_agents):
                
                grad_f[i, t] = f[i].g(x_list[i])
                x_new[i] = x_list[i] - gamma*(f[i].g(x_list[i]) + y[i] + z[i])
                x_old[i] = x_list[i]


            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents
                # print("agent %d: " % i)
                x_list[i] = R[cluster][ic, ic]*(x_new[i])
                for j in N_in[cluster][ic]:
                    x_list[i] += R[cluster][ic, j]*(x_new[cluster*nc_agents + j])


                # y_tmp[:, i, t+1] = R[cluster][ic, ic]*(y_tmp[:, i, t] + y[i] + z[i])
                # for j in N_in[cluster][ic]:
                #     y_tmp[:, i, t+1] += R[cluster][ic, j]*(y_tmp[:, cluster*nc_agents + j, t] + y[cluster*nc_agents + j] + z[cluster*nc_agents + j])
                
                # y_tmp[:, i, t+1] = R[cluster][ic, ic]*(y_tmp[:, i, t] + y[i])
                # for j in N_in[cluster][ic]:
                #     y_tmp[:, i, t+1] += R[cluster][ic, j]*(y_tmp[:, cluster*nc_agents + j, t] + y[cluster*nc_agents + j])
                
                # gf_mix[:, i, t+1] = R[cluster][ic, ic]*(gf_mix[:, i, t] + grad_f[i, t])
                # for j in N_in[cluster][ic]:
                #     gf_mix[:, i, t+1] += R[cluster][ic, j]*(gf_mix[:, cluster*nc_agents + j, t] + grad_f[cluster*nc_agents + j, t])
                
                # z_tmp[:, i, t+1] = R[cluster][ic, ic]*(z_tmp[:, i, t] + z[i])
                # for j in N_in[cluster][ic]:
                #     z_tmp[:, i, t+1] += R[cluster][ic, j]*(z_tmp[:, cluster*nc_agents + j, t] + z[cluster*nc_agents + j])
                
                # x_mix[:, i, t+1] = R[cluster][ic, ic]*(x_mix[:, i, t])
                # for j in N_in[cluster][ic]:
                #     x_mix[:, i, t+1] += R[cluster][ic, j]*(x_mix[:, cluster*nc_agents + j, t])
            


            # z_sum = [0 for i in range(n_agents)]
            # for i in range(n_agents):
            #     z_sum[i] = R[cluster][ic, ic]*(z[i])
            #     for j in N_in[cluster][ic]:
            #         z_sum[i] += R[cluster][ic, j]*(z[cluster*nc_agents + j])


            for i in range(n_agents):
                # if t != T-1:
                # z[i] = R[cluster][ic, ic]*(grad_f[i, t] + y[i])
                # for j in N_in[cluster][ic]:
                #     z[i] += R[cluster][ic, j]*(grad_f[cluster*nc_agents + j, t] + y[cluster*nc_agents + j])
                # z[i] = z[i] - grad_f[i, t] - y[i]
                # if t == 0:
                cluster = i//nc_agents
                ic = i % nc_agents
                z_r = R[cluster][ic, ic]*(x_new[i] - x_old[i] + gamma*y[i])
                # z_r = R[cluster][ic, ic]*(x_new[i] - x_old[i])
                for j in N_in[cluster][ic]:
                    z_r += R[cluster][ic, j]*(x_new[cluster*nc_agents + j] - x_old[cluster*nc_agents + j] + gamma*y[cluster*nc_agents + j])
                z_new[i] += 1/(T*gamma)*(x_new[i] - x_old[i] + gamma*y[i] - z_r)

                if t == T:
                    # z[i] = z[i] + 1/(gamma)*(x_new[i] - x_list[i])
                    z[i] = (gf_mix[:, i, 1] - (gf_mix[:, i, 0] + grad_f[i, 0]))\
                        + (y_tmp[:, i, 1] - y[i]) + z_tmp[:, i, 1]\
                        # + 1/gamma*(x_mix[:, i, 0] - x_mix[:, i, 1])
            
        # pdb.set_trace()
        
                # z[i] = z[i] - z_sum[i] + 1/(gamma)*(x_new[i] - x_list[i])

        if control == 1:
            sample_list = [sample_num_opt for i in range(opt.n_cluster)]
            com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), sample_list[i]) for i in range(opt.n_cluster)]
            com_list = [item for sublist in com_list for item in sublist]
        elif control == 2:
            com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
            com_list = [item for sublist in com_list for item in sublist]
        else:
            com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
            com_list = [item for sublist in com_list for item in sublist]

        # test = 0
        # y_tmp_sum = torch.mean(y_tmp, dim=2)
        # y_tmp_sum = y_tmp[:, :, -1]/T
        # z_tmp_sum = z_tmp[:, :, -1]/T

        # # y_tmp_sum2 = [(y_tmp[:, i, -2] + y[i] + z[i])/T for i in range(n_agents)]
        # y_tmp_sum2 = [(y_tmp[:, i, -2] + y[i])/T for i in range(n_agents)]

        # z_tmp_sum2 = [(z_tmp[:, i, -2] + z[i])/T for i in range(n_agents)]


        # for i in range(n_agents):
        #     z[i] = (gf_mix[:, i, 1] - (gf_mix[:, i, 0] + grad_f[i, 0]))\
        #         + (y_tmp[:, i, 1] - y[i]) + z_tmp[:, i, 1]\


        # pdb.set_trace()

        # for i in range(n_agents):
        #     z[i] = z[i] + 1/(gamma*T)*(x_new[i] - x_list[i])
        #     # y[i] = y[i] + 1/(gamma*T)*(x_new[i] - x_list[i])

        # for i in range(n_agents):
        #     # x_list[i] = R[cluster][ic, ic]*(x_new[i] + T*gamma*(z_tmp_sum2[i]))
        #     # for j in N_in[cluster][ic]:
        #     #     x_list[i] += R[cluster][ic, j]*(x_new[cluster*nc_agents + j] + T*gamma*(z_tmp_sum2[cluster*nc_agents + j]))
        #     # z[i] = z_tmp_sum2[i] + 1/(T*gamma)*(x_new[i] - x_list[i])
        #     # z[i] = R[cluster][ic, ic]*(grad_f[i, -1])
        #     # for j in N_in[cluster][ic]:
        #     #     z[i] += R[cluster][ic, j]*(grad_f[cluster*nc_agents + j, -1])
        #     # z[i] = z[i] - grad_f[i, -1]

        #     z[i] = z[i] + 1/(gamma)*(x_new[i] - x_list[i])
        #     # z[i] = 1/(gamma)*(x_mix[:, i, -2] - x_mix[:, i, -1]) + (gf_mix[:, i, -1] - (gf_mix[:, i, -2] + grad_f[i, -1])) \
        #     #     + (z_tmp[:, i, -1] - (z_tmp[:, i, -2] + z[i]))\
        #     #     + (y_tmp[:, i, -1] - (y_tmp[:, i, -2] + y[i]))\
        #     #     + z[i]


        # x_cen = 0
        y_cen_tmp = 0
        mean_grad = 0
        mean_z = 0
        x_mix_sum = 0
        x_cen_delta = 0
        
        delta_x = [0 for i in range(n_agents)]
        for i in range(n_agents):
            delta_x[i] = x_list[i] - x_mix[:, i, 0] + T*gamma*y[i]
            # delta_x[i] = x_list[i] - x_mix[:, i, 0] 

            # delta_x[i] = x_list[i]

        cluster_x = [0 for i in range(n_agents//nc_agents)]
        for i in range(n_agents):
            cluster = i//nc_agents
            ic = i % nc_agents
            if i in com_list:
                # print("cluster: %d, i: %d" % (cluster, i))
                if control == 1:
                    cluster_x[cluster] += 1/sample_list[cluster]*delta_x[i]
                else:
                    cluster_x[cluster] += 1/opt.sample_num*delta_x[i]
                # cluster_y[cluster] += 1/opt.sample_num*y[i]
        
        # pdb.set_trace()

        for n_c in com_list:
            cluster = n_c//nc_agents
        # for n_c in range(n_agents):
            # x_cen += 1/len(com_list)*(x_list[n_c] + T*gamma*(y[n_c]))
            # x_cen += 1/len(com_list)*(x_list[n_c] + T*gamma*(y_tmp_sum[:, n_c]))
            # x_cen += 1/len(com_list)*(x_list[n_c] + T*gamma*(y[n_c]))
            if control == 1:
                x_cen_delta += 1/(opt.n_cluster*sample_list[cluster])*(delta_x[n_c])
            else:
                x_cen_delta += 1/len(com_list)*(delta_x[n_c])
            # x_cen2 += 1/len(com_list)*(x_list[n_c])


            # x_cen += 1/len(com_list)*(x_new[n_c] + T*gamma*(y_tmp_sum2[n_c]))
            # mean_grad += 1/len(com_list)*(torch.mean(grad_f[n_c], dim = 0))
            # mean_grad += 1/len(com_list)*(gf_mix[:, n_c, -1]/T)

            # mean_grad += 1/len(com_list)*((gf_mix[:, n_c, -2] + grad_f[n_c, -1])/T)
            
            # mean_z += 1/len(com_list)*(z_tmp[:, n_c, -2] + z[n_c])/T
            # mean_z += 1/len(com_list)*z_tmp_sum2[n_c]


            # x_mix_sum += 1/len(com_list)*(x_mix[:, n_c, -1])
            # x_mix_sum += 1/len(com_list)*(x_mix[:, n_c, -2])


            # x_cen += 1/len(com_list)*(x_new[n_c])

            # y_cen_tmp += 1/len(com_list)*(x_new[n_c] + T*gamma*(y_tmp_sum2[n_c]) - y_cen)
            # c[n_c] = x_new[n_c] + T*gamma*(y_tmp_sum2[n_c])
        # y_cen = y_cen + len(com_list)/n_agents*y_cen_tmp

        # x_cen_new = 0
        # for n_c in com_list:
        #     x_cen_new += 1/len(com_list)*(x_list[n_c] + T*gamma*(y_tmp_sum[:, n_c]))
        # x_cen = rho_m*x_cen + (1 - rho_m)*x_cen_new
        for i in range(n_agents//nc_agents):
            # cluster_y[i] = cluster_y[i] + 1/(T*gamma)*(cluster_x[i] - x_cen_delta)
            cluster_y[i] = 1/(T*gamma)*(cluster_x[i] - x_cen_delta)

        for i in range(n_agents):
            cluster = i//nc_agents
            if i in com_list:
                # y[i] = cluster_y[cluster] + 1/(T*gamma)*(cluster_x[cluster] - x_cen_delta)
                y[i] = cluster_y[cluster]


        # pdb.set_trace()

        for i in range(n_agents):
            z[i] = z_new[i]
        '''broadcast'''

        # error_z.append(torch.norm(x_list[0] - x_cen).item())
        for i in range(n_agents):
            y_old[i] = y[i]

        x_cen += x_cen_delta
        for i in range(n_agents):
            cluster = i//nc_agents
            ic = i % nc_agents
            if i in com_list:
                x_list[i] = x_cen

            

        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(x_cen.reshape(dim, 1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        if control == 1:
            total = 0
            total += np.mean(energy_list)*opt.delta*T
            for i in range(opt.n_cluster):
                # total += ((sample_list[i]/nc_agents)**2*energy_list[i])/opt.n_cluster
                total += ((sample_list[i]/nc_agents)*energy_list[i])/opt.n_cluster

            energy_count += total
        elif control == 2:
            energy_count += np.mean(energy_list)*opt.delta*T
            energy_count += np.mean(energy_list)
        energy_cost_list.append(energy_count)
        # pdb.set_trace()

        # cluster = 0//nc_agents
        # ic = 0 % nc_agents
        # x_c = torch.mean(torch.stack(x_list[:nc_agents]), dim = 0)
        # z_prox = (f[0].g(x_cen) + y[0])
        # # pdb.set_trace()
        # zy_prox = f[0].g(x_cen)
        # y_prox = 0
        # dy_prox = y[0] - y_old[0]
        # for j in range(nc_agents):
        #     z_prox -= 1/(nc_agents)*(f[j].g(x_cen) + y[j])
        #     # z_prox -= R[0][ic, j]*(grad_f[j, 0] + y[j])
        #     y_prox += 1/(nc_agents)*(f[j].g(x_cen) + y[j])
        #     dy_prox -= 1/(nc_agents)*(y[j] - y_old[j])


        # for j in range(n_agents):
        #     zy_prox -= 1/n_agents*(f[j].g(x_cen))
        #     y_prox -= 1/n_agents*(f[j].g(x_cen))

        # error_z.append(torch.norm(dy_prox).item())




        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % opt.p_iter == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # print("norm: %.2E" % error_z[-1], '\t\t\t\t\r')

            # pdb.set_trace()
    if control >= 1:
        return error_list, error_z, energy_cost_list
    return error_list, error_z


def Gradient_Tracking(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1,  opt=None):
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

    test = 0

    # y = [0 for i in range(n_agents)]
    y = [0 for i in range(n_agents)]
    yt = [0 for i in range(n_agents)]

    z = [0 for i in range(n_agents)]

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
    # rho_m *= 3
    # pdb.set_trace()
    # rho_m = 1 - (nc_agents - opt.sample_num)/nc_agents
    # W = torch.block_diag(*torch.stack(R))
    # Big_J = torch.ones(n_agents, n_agents)/n_agents

    T = int(opt.p_inv)
    z = [0 for i in range(n_agents)]

    for r in range(int(opt.c_rounds)):
        # y_tmp = [0 for i in range(n_agents)]
        y_tmp = torch.zeros((opt.dim, n_agents, T+1), dtype=torch.double).to(x.device)
        grad_f = torch.zeros((n_agents, T, opt.dim), dtype=torch.double).to(x.device)
        x_init = torch.stack(x_list)

        for t in range(T):
            for i in range(n_agents):

                grad_f[i, t] = f[i].g(x_list[i])
                x_new[i] = x_list[i] - gamma*(y[i])

                # pdb.set_trace()
                # pdb.set_trace()

            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents
                # print("agent %d: " % i)
                x_list[i] = R[cluster][ic, ic]*(x_new[i])
                for j in N_in[cluster][ic]:
                    x_list[i] += R[cluster][ic, j]*(x_new[cluster*nc_agents + j])

                # y_tmp[:, i, t+1] = R[cluster][ic, ic]*(y_tmp[:, i, t] + y[i] + z[i])
                # for j in N_in[cluster][ic]:
                #     y_tmp[:, i, t+1] += R[cluster][ic, j]*(y_tmp[:, cluster*nc_agents + j, t] + y[cluster*nc_agents + j] + z[cluster*nc_agents + j])

                # z[i] = z[i] + 1/(gamma*T)*(x_new[i] - x_list[i])
                yt[i] = R[cluster][ic, ic]*(y[i])
                for j in N_in[cluster][ic]:
                    yt[i] += R[cluster][ic, j]*(y[cluster*nc_agents + j])
            
            # if t != T-1:
            for i in range(n_agents):
                y[i] = yt[i] + f[i].g(x_list[i]) - grad_f[i, t]
            # pdb.set_trace()
        com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        com_list = [item for sublist in com_list for item in sublist]

        x_cen = 0
        y_cen = 0
        # test = 0
        # y_tmp_sum = torch.mean(y_tmp, dim=2)

        # '''one more update'''
        # for i in range(n_agents):
        #     x_list[i] = x_list[i] - gamma*(f[i].g(x_list[i]) + y[i])
        #     # grad_f[i, t] = f[i].g(x_list[i])

        mean_grad_f = torch.mean(grad_f, dim=1)
        # pdb.set_trace()
        for n_c in com_list:
            # x_cen += 1/len(com_list)*(x_list[n_c] + T*gamma*(y[n_c]))
            x_cen += 1/len(com_list)*(x_list[n_c])
            y_cen += 1/len(com_list)*(yt[n_c])

            # x_cen += 1/len(com_list)*(x_list[n_c] + T*gamma*(y[n_c]))

            # test += 1/len(com_list)*(x_new[n_c])
        # pdb.set_trace()
        # pdb.set_trace()

        # pdb.set_trace()

        '''broadcast'''

        # avg_y = [0 for i in range(opt.n_cluster)]

        for i in range(n_agents):
            if i in com_list:
                # y[i] = y_tmp_sum[:, i] + 1/(T*gamma)*(x_list[i] - x_cen)
                cluster = i//nc_agents
                ic = i % nc_agents

                x_list[i] = x_cen
                y[i] = y_cen + f[i].g(x_list[i]) - grad_f[i, -1]


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


def SCAFFOLD(x, input, labels, R, C, N_in, N_out, x_opt, fstar, lam=0, T=1, n_agents=10, L=10, step_c=1, control=0, energy_list=None, opt=None):
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

    # x = torch.zeros(dim).to(x.device)
    x_list = [x.type(torch.double) for i in range(n_agents)]

    com_count = 0
    gamma = torch.tensor(step_c*1, dtype=torch.double)

    error_list = []

    energy_cost_list = []

    energy_count = 0

    error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1))/n_agents).item())

    com_count = 0
    com_iter = 0
    fg_old = torch.zeros(n_agents, dim).type(torch.double).to(x.device)
    iters = 0
    print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    # while (com_count < opt.c_rounds):
    nc_agents = n_agents//opt.n_cluster

    J = torch.ones(nc_agents, nc_agents)/nc_agents
    # pdb.set_trace()
    # rho_m *= 3
    # pdb.set_trace()
    # rho_m = 1 - (nc_agents - opt.sample_num)/nc_agents
    # W = torch.block_diag(*torch.stack(R))
    # Big_J = torch.ones(n_agents, n_agents)/n_agents

    T = int(opt.p_inv)

    x_cen = x
    c_cen = 0
    c = [0 for i in range(n_agents)]
    c_new = [0 for i in range(n_agents)]



    sample_count = [1 for i in range(n_agents)]

    
    error_z = []
    # energy_list = [random.randint(1, 200) for i in range(opt.n_cluster)]

    for r in range(int(opt.c_rounds)):
        com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        com_list = [item for sublist in com_list for item in sublist]
        for n_c in com_list:
            x_list[n_c] = x_cen

        for t in range(T):
            for i in com_list:
                x_list[i] = x_list[i] - gamma*(f[i].g(x_list[i]) - c[i] + c_cen)
        
        for i in com_list:
            c_new[i] = c[i] - c_cen + 1/(T*gamma)*(x_cen - x_list[i])

        delta_x = torch.mean(torch.stack([x_list[i] - x_cen for i in com_list]), dim = 0)
        delta_y = torch.mean(torch.stack([c_new[i] - c[i] for i in com_list]), dim=0)
        for i in com_list:
            c[i] = c_new[i]
        
        x_cen = x_cen + delta_x
        c_cen = c_cen + len(com_list)/n_agents*delta_y
        

        iters += 1
        com_iter += 1
        com_count += 1
        if x_list[0].dtype != torch.float64:
            pdb.set_trace()
        error_list.append((torch.norm(x_cen.reshape(dim, 1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

        if iters % opt.p_iter == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # print("norm: %.2E" % error_z[-1], '\t\t\t\t\r')

            # pdb.set_trace()
    return error_list, error_z
