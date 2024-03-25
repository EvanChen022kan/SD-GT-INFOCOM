import torch
import numpy as np
from .gradfn import func
from .utils import get_mse_fstar
import pdb


def ProxSkip(x, input, labels, fstar, x_opt=0, c_rounds = 1, lam=0, p=1.0, n_agents=10, L=10, step_c = 1, opt=None):
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

    diff = []

    f = []
    h = []
    for (inp, lab) in zip(input_list, label_list):
        f_i = func(inp, lab, lam = lam, opt=opt)
        f.append(f_i)
        h.append(torch.zeros(x.shape[0]).to(x.device))
    x_list = [x for i in range(n_agents)]

    def global_f(x): return torch.mean(torch.stack([f[i].f(x) for i in range(n_agents)]))
    def global_g(x): return torch.mean(torch.stack([f[i].g(x) for i in range(n_agents)], dim=0), dim=0)
    com_count = 0

    # gamma = 1/L
    gamma = step_c*1

    # x_opt = x.clone()
    error_list = []
    # error_list.append((global_f(x) - fstar).item()) # save the initial gap
    error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())
    com_count = 0
    com_iter = 0
    iters = 0
    while (com_count < c_rounds):
        theta_t = np.random.binomial(1, p)
        # x_opt = x_opt - gamma*global_g(x_opt)

        for i in range(n_agents):
            fg = f[i].g(x_list[i])
            t_fg = global_g(x_list[i])
            x_list[i] = x_list[i] - gamma*(fg - h[i])
        iters += 1
        com_iter += 1
        # if theta_t == 1:
        # diff.append(torch.norm(x_opt - x_list[0]).item())
        diff.append(torch.norm(t_fg - (fg- h[-1])).item())


        if iters % int(1/p) == 0:
            # print("[%d agents]Communication Count: " % n_agents, com_count, '\tIters:', iters, '\t\t\t\t\r', end='')
            x_hat = torch.mean(torch.stack(x_list), dim=0)
            

            # x_opt = x_hat
            # x_hat = x_opt

            for i in range(n_agents):
                h[i] = h[i] + p/gamma*(x_hat - x_list[i])
                # h[i] = h[i] + 1/(gamma*com_iter)*(x_hat - x_list[i])

                x_list[i] = x_hat
            

            com_count += 1
            com_iter = 0
            # error_list.append((global_f(x_hat) - fstar).item())
            error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.reshape(dim, 1), 'fro')/n_agents).item())

            if com_count % int(0.25*opt.c_rounds) == 0:
                # print("[%d agents]Communication Count: " % n_agents, com_count, '\t||Error: %.2E' % error_list[-1].item(), '\t\t\t\t\r', end='')
                print("[%d agents]Communication Count: " % n_agents, com_count, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
                # print("%.10f" % global_f.f(x_hat))
        
        # if iters % 100 == 0:
        #     x_hat = torch.mean(torch.stack(x_list), dim=0)
        #     error_list.append((global_f.f(x_hat) - fstar).item())
    # pdb.set_trace()
    if opt.test_diff:
        return diff, error_list
    return error_list


