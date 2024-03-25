import torch
import numpy as np
from .gradfn import func
from .utils import update_fstar
import pdb


def NewProx(x, input, labels, fstar, lam=0, p=1.0, n_agents=10, L=10, LR_c = 1, beta = 1, method = 1, opt = None):
    N = input.shape[0]
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    label_list = list(torch.split(labels, n, dim=0))
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)
    

    f = []
    h = []
    h_old = []
    grad_old = []
    for (inp, lab) in zip(input_list, label_list):
        f_i = func(inp, lab, lam = lam, opt=opt)
        f.append(f_i)
        h.append(torch.zeros(x.shape[0]).to(x.device))

        h_old.append([])
        grad_old.append([])
    x_list = [x for i in range(n_agents)]
    def global_f(x): return torch.mean(torch.stack([f[i].f(x) for i in range(n_agents)]))
    def global_g(x): return torch.mean(torch.stack([f[i].g(x) for i in range(n_agents)], dim = 0), dim = 0)

    grad = [[] for i in range(n_agents)]

    # com_count = 0
    # gamma = LR_c*1/L
    # gamma2 = 1/(L*n_agents)
    # gamma1 = p/L
    gamma1 = p/L
    gamma2 = p/L
    # gamma1 = 1/(n_agents*L)
    # gamma2 = 1/(n_agents*L)

    # pdb.set_trace()



    error_list = []
    diff = []
    error_list.append((global_f(x) - fstar).item())  # save the initial gap
    com_count = 0
    iters = 0
    com_iter = 0
    # fg_list = [torch.zeros(opt.dim).to(x.device) for i in range(n_agents)]
    fg_list = [[] for i in range(n_agents)]

    z_list = [torch.zeros(opt.dim).to(x.device) for i in range(n_agents)]
    z = torch.zeros(opt.dim).to(x.device)
    # z_list = [f[i].g(x_list[i]) for i in range(n_agents)]
    # z = torch.sum(torch.stack(z_list, dim = 0), dim = 0)

    x_opt = x.clone()
    # print("[%d agents][prob: %.1E]Communication Count: " % (n_agents, p), com_count, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    x_hat = x.clone()
    while (com_count < opt.c_rounds):
        theta_t = np.random.binomial(1, p)
        # t_fg = global_g(x_opt)
        x_opt = x_opt - 1/L*global_g(x_opt)

        for i in range(n_agents):
            fg = f[i].g(x_list[i])
            t_fg = global_g(x_list[i])
            fg_list[i].append(fg)
            if com_count == 0:
                x_list[i] = x_list[i] - gamma1*fg
                # x_list[i] = x_list[i] - gamma*(1/n_agents*fg + (z - z_list[i]))
            else:
                x_list[i] = x_list[i] - gamma1*(z + beta*(1/n_agents*fg - z_list[i]))
                # pdb.set_trace()

            # if com_iter == 0:
            #     pdb.set_trace()
        # t1 = 1/n_agents*torch.sum(torch.stack(fg_list, dim=0), dim=0)
        # t2 = global_g(x_hat)
        # pdb.set_trace()
        iters += 1 
        com_iter += 1
        # print('Iters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')


        # if theta_t == 1:
        # diff.append(torch.norm(x_opt - x_list[0]).item())
        if com_count == 0:
            diff.append(torch.norm(t_fg - (fg_list[-1][-1])).item())
        else:
            diff.append(torch.norm(t_fg - (z + beta*(1/n_agents*fg_list[-1][-1] - z_list[-1]))).item())

        if iters % int(1/p) == 0:
            # h_sum = [torch.sum(torch.stack(h_old[i]), dim=0) for i in range(n_agents)]
            # x_hat = torch.mean(torch.stack([x_list[i] - gamma*h_sum[i] for i in range(n_agents)]), dim=0) # jitter problem
            # x_hat = torch.mean(torch.stack(x_list, dim = 0), dim = 0)  # jitter problem

            for i in range(n_agents):
                # z_list[i] = 1/n_agents*p*fg_list[i]
                if method == 2:
                    z_list[i] = 1/n_agents*fg_list[i][0]
                else:
                    z_list[i] = 1/n_agents*torch.mean(torch.stack(fg_list[i], dim=0), dim=0)
                # z_list[i] = 1/n_agents*fg_list[i][0]
                # x_list[i] = x_hat
                # z_list[i] = p/gamma1*(x_hat - x_list[i])
            # pdb.set_trace()
            # t1 = [torch.tensor([1, 1, 1]) for i in range(5)]
            # t2 = torch.sum(torch.stack(t1, dim=0), dim=0)
            # pdb.set_trace()
            z = torch.sum(torch.stack(z_list, dim = 0), dim = 0)
            z2 = torch.mean(torch.stack([torch.mean(torch.stack(fg_list[i], dim=0), dim=0) for i in range(n_agents)], dim=0), dim=0)
            # pdb.set_trace()
            x_hat = x_hat - 1/p*gamma2*(z2)
            # x_hat = x_opt
            # x_hat2 = torch.mean(torch.stack(x_list, dim=0), dim=0)
            # pdb.set_trace()
            # z2 = torch.mean(torch.stack([torch.mean(torch.stack(fg_list[i], dim=0), dim=0) for i in range(n_agents)], dim=0), dim=0)
            # x_hat = x_hat - 1/p*1/L*(z2)


            # for i in range(n_agents):
            #     z_list[i] = 1/n_agents*f[i].g(x_hat)
            # z = torch.sum(torch.stack(z_list, dim = 0), dim = 0)

            # x_opt = x_hat
            # x_hat = x_opt

            x_list = [x_hat for i in range(n_agents)]

            # fg_list = [torch.zeros(opt.dim).to(x.device) for i in range(n_agents)]
            fg_list = [[] for i in range(n_agents)]


            com_count += 1
            error_list.append((global_f(x_hat) - fstar).item())
            
            com_iter = 0
            
            if global_f(x_hat) < fstar:
                fstar = global_f(x_hat).item()
                update_fstar(fstar - 1e-14, x_hat, n_agents)
                print("fstar updated: %.2E" % fstar)
            # print("[%d agents][prob: %.1E]Communication Count: " % (n_agents, p), com_count, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

            if int(0.1*opt.c_rounds) > 0 and com_count % int(0.1*opt.c_rounds) == 0:
                # print("[%d agents]Communication Count: " % n_agents, com_count, '\t||Error: %.2E' % error_list[-1].item(), '\t\t\t\t\r', end='')
                print("[%d agents][prob: %.1E]Communication Count: " % (n_agents, p), com_count, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    if opt.test_diff:
        return diff, error_list
    return error_list

