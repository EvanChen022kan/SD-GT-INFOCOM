import torch
import numpy as np
from .gradfn import func
import pickle
from .utils import update_fstar, get_fstar, get_mse_fstar
import pdb


def GetOpt(x, input, labels, lam=0, L=10, opt=None):
    N = input.shape[0]
    # global_f = func(input, labels, lam=lam)

    x_opt = x
    com_count = 0
    gamma = 1/L

    n_agents = opt.n_agents
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    label_list = list(torch.split(labels, n, dim=0))
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)

    fstar, _ = get_mse_fstar(input_list, label_list, lam)
    print(fstar.item())

    x_list = [x for i in range(n_agents)]

    f = []
    h = [torch.zeros(x.shape[0]).to(x.device) for i in range(n_agents)]
    # for (inp, lab) in zip(input_list, label_list):
    for i in range(n_agents):
        f_i = func(input_list[i], label_list[i], lam=lam, opt=opt)
        f.append(f_i)

    def global_f(x): return torch.mean(torch.stack([f[i].f(x) for i in range(n_agents)]))
    def global_g(x): return torch.mean(torch.stack([f[i].g(x) for i in range(n_agents)], dim = 0), dim = 0)
    f_star, _ = get_fstar(n_agents)

    com_count = 0
    iters = 0
    # pdb.set_trace()
    print("[%d][%d agents] %.13f" % (com_count, n_agents, global_f(x_opt)), '\t\t\t\r')
    while (com_count < 10000000):
        # for i in range(n_agents):
        #     x_list[i] = x_list[i] - gamma*(f[i].g(x_list[i]) - h[i])
        # if com_count % 1000 == 0:
        #     x_hat = torch.mean(torch.stack(x_list), dim=0)
        #     for i in range(n_agents):
        #         h[i] = h[i] + 1e-3/gamma*(x_hat - x_list[i])
        #         x_list[i] = x_hat

        x_opt = x_opt - gamma*global_g(x_opt)

        if com_count % 1000 == 0:
            print("[%d][%d agents] %.13f" % (com_count, n_agents, global_f(x_opt)), '\t\t\t\r', end='')
            # print("[%d][%d agents] %.13f" % (com_count, n_agents, global_f(x_opt)), '\t\t\t\r')
            if global_f(x_opt).item() < f_star:
                f_star = global_f(x_opt).item()
                update_fstar(f_star, x_opt, n_agents)
                print("\nupdated")
        com_count += 1

    return global_f(x_opt).item()
    pdb.set_trace()
