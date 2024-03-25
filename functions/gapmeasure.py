import torch
import numpy as np
from .gradfn import func
from .utils import update_fstar
import pdb


def GapMeasure(x, input, labels, fstar, lam=0, p=1.0, n_agents=10, L=10, LR_c = 1, opt=None):
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
    h2 = []
    h_old = []
    grad_old = []
    grad = []

    for (inp, lab) in zip(input_list, label_list):
        f_i = func(inp, lab, lam = lam, opt=opt)
        f.append(f_i)
        h.append(torch.zeros(x.shape[0]).to(x.device))
        h2.append(torch.zeros(x.shape[0]).to(x.device))
        h_old.append([])
        grad_old.append([])
        grad.append([])


    x_list = [x for i in range(n_agents)]
    x_list2 = [x for i in range(n_agents)]
    x_list3 = [x for i in range(n_agents)]

    def global_f(x): return torch.mean(torch.stack([f[i].f(x) for i in range(n_agents)]))
    def global_g(x): return torch.mean(torch.stack([f[i].g(x) for i in range(n_agents)], dim = 0), dim = 0)


    # global_f = func(input, labels, lam=lam)
    com_count = 0
    gamma = LR_c*1/L
    # eta_l = min(1/(81*L*(1/p)*1), 1/(15*lam*(1/p)*1))
    eta_l = 1/L



    com_list = []
    com_list2 = []
    com_list3 = []


    # error_list.append((global_f(x) - fstar).item())  # save the initial gap
    x_opt = x.clone()
    iters = 0
    com_iter = 0


    observe = []
    c_loc = [torch.zeros(x.shape[0]).to(x.device) for i in range(n_agents)]
    c_g = torch.zeros(x.shape[0]).to(x.device)
    x_hat2 = x_hat = x_hat3 = x
    print("[%d agents]Communication Count: " % n_agents, com_count, '\tIters:', iters, "\tProx: %.2E, Ours: %.2E" % ((global_f(x_hat) - fstar).item(), (global_f(x_hat3) - fstar).item()))



    while (com_count < opt.c_rounds):

        '''get opt sequence'''
        x_opt = x_opt - gamma*global_g(x_opt)
        h_opt = f[0].g(x_list[0]) - global_g(x_opt)


        theta_t = np.random.binomial(1, p)

        for i in range(n_agents):
            x_list[i] = x_list[i] - gamma*(f[i].g(x_list[i]) - h[i])
            x_list2[i] = x_list2[i] - eta_l*(f[i].g(x_list2[i]) - c_loc[i] + c_g)


            fg = f[i].g(x_list3[i])
            if not len(grad_old[i]) == 0:
                if com_iter < len(grad_old[i]):
                    h2[i] = h2[i] - p*(n_agents-1)/n_agents*grad_old[i][com_iter] + p*(n_agents-1)/n_agents*fg
                    # h2[i] = h2[i] - p*(n_agents-1)/n_agents*torch.mean(torch.stack(grad_old[i]), dim=0) + p*(n_agents-1)/n_agents*fg
            x_list3[i] = x_list3[i] - gamma*(fg - h2[i])

            h_old[i].append(h2[i])
            grad[i].append(fg)
            # if not len(grad_old[i]) == 0:
            #     if com_iter < len(grad_old[i]):
            #         h2[i] = h2[i] - p*(n_agents-1)/n_agents*grad_old[i][com_iter] + p*(n_agents-1)/n_agents*fg
                # else:
                #     h2[i] = h2[i] - p*(n_agents-1)/n_agents*torch.mean(torch.stack(grad_old[i]), dim=0) + p*(n_agents-1)/n_agents*fg


        iters += 1
        com_iter += 1

        # if theta_t == 1:
        if iters % int(1/p) == 0:
            
            x_hat = torch.mean(torch.stack(x_list), dim=0)
            # pdb.set_trace()
            h_sum = [torch.sum(torch.stack(h_old[i]), dim=0) for i in range(n_agents)]
            # h_mean = [torch.mean(torch.stack(h_old[i]), dim=0) for i in range(n_agents)]

            x_hat3 = torch.mean(torch.stack([x_list3[i] - gamma*h_sum[i] for i in range(n_agents)]), dim=0)
            # x_hat3 = torch.mean(torch.stack([x_list3[i] - gamma/p*h2[i] for i in range(n_agents)]), dim=0)
            # x_hat3 = torch.mean(torch.stack(x_list3), dim=0)
            # pdb.set_trace()
            delta = []
            for i in range(n_agents):
                h[i] = h[i] + p/gamma*(x_hat - x_list[i])
                h2[i] = h2[i] + p/gamma*(x_hat3 - x_list3[i])
                # h2[i] = p*h_sum[i] + p/gamma*(x_hat3 - x_list3[i])


                delta_c = -c_g + p/eta_l*(x_hat2 - x_list2[i])
                c_loc[i] = c_loc[i] + delta_c
                delta_y = x_list2[i] - x_hat2
                delta.append(torch.stack([delta_c, delta_y]))

                x_list[i] = x_hat
                x_list3[i] = x_hat3

            delta = torch.mean(torch.stack(delta, dim=0), dim = 0)

            x_hat2 = x_hat2 + delta[1]
            c_g = c_g + delta[0]
            for i in range(n_agents):
                x_list2[i] = x_hat2

            com_iter = 0
            com_count += 1

            grad_old = grad
            grad = [[] for i in range(n_agents)]
            h_old = [[] for i in range(n_agents)]
            com_list.append((global_f(x_hat) - fstar).item())
            com_list3.append((global_f(x_hat2) - fstar).item())
            # if global_f(x_hat3) < fstar:
            #     fstar = global_f(x_hat3).item()
            #     update_fstar(fstar - 1e-14, x_hat3, n_agents)
            #     print("fstar updated: %.2E" % fstar)

            com_list2.append((global_f(x_hat3) - fstar).item())


            print("[%d agents]Communication Count: " % n_agents, com_count, '\tIters:', iters, "\tProx: %.2E, Ours: %.2E" % ((global_f(x_hat) - fstar).item(), (global_f(x_hat3) - fstar).item()))
            # com_list2.append((global_f(x_hat3) - fstar).item())
            # observe.append(global_f(x_hat).item() - global_f(x_opt).item())
            observe.append(torch.norm(global_g(x_hat) - global_g(x_opt)).item())


            



        # for i in range(n_agents):
        #     prox_gap_list[i].append(torch.norm(x_list[i] - x_opt).item())
        #     gap_list[i].append(torch.norm(x_list3[i] - x_opt).item())
    # pdb.set_trace()
    
    return [observe, com_list3, com_list, com_list2]
