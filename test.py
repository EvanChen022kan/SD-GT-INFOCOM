from libsvm import svmutil
import numpy as np
import torch
from functions import ProxSkip, NewProx, save_img, GapMeasure, GetOpt, get_fstar, update_fstar, del_fstar, get_L, get_mse_fstar, get_chain_graph, get_circle_graph
# from functions.SONATA import ASYSONATA, OLD_ASYSONATA, NEW_ASYSONATA, ASY_GT, ASY_SGD
from functions.SemiFL import SemiFL_GT, SemiFL, SemiFL_GT2, SemiFL_GT3, SemiFL_ScatterGT, SemiFL_ScatterGT2
from functions.graph_gen import get_FC_graph, get_ring_graph, get_grid_graph, get_2star_graph
from functions.RandomSkip import RandomSkip, Skip2
from functions.test import Test1, Test2
import matplotlib.pyplot as plt
import matplotlib
from settings import Option
import pickle
import scienceplots
import random
import scipy.io as sio
import time
import pdb

matplotlib.use('Agg')
torch.manual_seed(123456)
# plt.yscale('log')
opt = Option("train")
gpu = torch.device('cuda:{}'.format(opt.gpu_id))
dim = opt.dim
n_agents = opt.n_agents


''' Load data and convert to numpy arrays'''
if opt.data_source != 'w8a':
    opt.dim = dim = 200
    N_per_agent = 30

    omega = opt.omega
    input = torch.randn((N_per_agent*n_agents, dim), dtype=torch.double).to(gpu)
    x_init = torch.randn((dim), dtype=torch.double).to(gpu)

    for i in range(input.shape[1]):
        if i == 0:
            input[:, i] = input[:, i]/np.sqrt((1 - omega**2))
        else:
            input[:, i] = omega*input[:, i-1] + input[:, i]

    noise = torch.randn((N_per_agent*n_agents), dtype=torch.double).to(gpu)
    labels = input @ x_init + 0.2*noise

    AAt = input.transpose(0, 1) @ input
    t0 = torch.real(torch.linalg.eig(AAt)[0])
    kappa = torch.max(t0)/torch.min(t0)
    A_L = torch.max(t0)
    print("(omega = %.2f)Kappa value: %.2f" % (omega, kappa))

n_cluster = opt.n_cluster
R = []
C = []
N_in = []
N_out = []

for i in range(n_cluster):
    if opt.use_ring and n_agents//n_cluster > 1:
        r, c, n_in, n_out = get_ring_graph(n_agents//n_cluster)
    elif opt.use_grid:
        r, c, n_in, n_out = get_grid_graph(n_agents//n_cluster)
    elif opt.use_2star:
        r, c, n_in, n_out = get_2star_graph(n_agents//n_cluster)
    else:
        r, c, n_in, n_out = get_FC_graph(n_agents//n_cluster)

    R.append(r)
    C.append(c)
    N_in.append(n_in)
    N_out.append(n_out)
# pdb.set_trace()

'''get fstar'''
if opt.loss == 'mse':
    N = input.shape[0]
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    label_list = list(torch.split(labels, n, dim=0))
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)

    mu = 0

    ''' get opt '''
    I = torch.eye(dim, dim).to(gpu)
    x_opt = torch.linalg.inv(input.transpose(0, 1) @ input) @ input.transpose(0, 1) @ labels

    f_star = 0
f_star -= 2e-17

'''get cluster x_opt'''
# I = torch.eye(dim, dim).to(gpu)
# c_val = n_agents//n_cluster
# x_opt = []
# for i in range(0, n_agents, c_val):
#     input_c = torch.cat(input_list[i:i+c_val], dim = 0)
#     label_c = torch.cat(label_list[i:i+c_val], dim=0)
#     x_opt_c = torch.linalg.inv(input_c.transpose(0, 1) @ input_c) @ input_c.transpose(0, 1) @ label_c
#     x_opt.append(x_opt_c)
# x_opt = torch.mean(torch.stack(x_opt, dim = 0), dim = 0)
'''get cross-cluster x_opt'''
# c_val = n_agents//n_cluster
# x_opt = []
# def all_choices(length, per_cluster):
#     output = []
#     if length != per_cluster:
#         sub_list = all_choices(length - per_cluster, per_cluster)
#         for i in range(per_cluster):
#             for combinations in sub_list:
#                 output.append([i] + [x+per_cluster for x in combinations])
#         return output
#     else:
#         return [[i] for i in range(per_cluster)]
# choices = all_choices(n_agents, c_val)
# # pdb.set_trace()
# for i, choice in enumerate(choices):
#     if i % 500 == 0:
#         print("Progress: %d/%d" % (i, len(choices)), "\t\t\t\t\r", end = '')
#     input_m = torch.cat([input_list[p] for p in choice], dim = 0)
#     label_m = torch.cat([label_list[p] for p in choice], dim=0)
#     x_opt_m = torch.linalg.inv(input_m.transpose(0, 1) @ input_m) @ input_m.transpose(0, 1) @ label_m
#     x_opt.append(x_opt_m)
# x_opt = torch.mean(torch.stack(x_opt, dim = 0), dim = 0)


x = torch.randn(dim, dtype=torch.double).to(gpu)


err_list1 = []
err_list2 = []
err_list3 = []

x_label1 = []
x_label2 = []
x_label3 = []

min_step1 = 3.5
min_err1 = 10
min_step2 = 3.5
min_err2 = 10
min_step3 = 3.5
min_err3 = 10

N = 10

prob_list = [1/int(np.clip(np.random.normal(opt.p_inv, 0.5*opt.p_inv, 1), 1, 3*opt.p_inv)) for i in range(n_agents)]
start_val = (1/A_L).item()


step_c_list = [((i+1)/N)*10*start_val for i in range(N-1)] + [((i+1)/N)*100
                                                              * start_val for i in range(N-1)] + [((i+1)/N)*1000
                                                                                                  * start_val for i in range(N-1)] + [((i+1)/N)*1e4*start_val for i in range(N-1)]

# step_c_list = [1.2*1e-5]
step_c_list = [opt.lr]


skip_list = [np.random.binomial(1, prob_list[i], int(opt.c_rounds)) for i in range(n_agents)]

for i in range(len(step_c_list)):
    step_c = step_c_list[i]
    print("Progress: %d/%d, step_c = %.2E" % ((i+1), len(step_c_list), step_c))
    err3 = Test2(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=A_L, step_c=step_c, opt=opt)
    err1 = Test1(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=A_L, step_c=step_c, opt=opt)

    if err1[-1] < err1[0]:
        err_list1.append(err1[-1])
        x_label1.append(step_c)
    # if err2[-1] < err2[0]:
    #     err_list2.append(err2[-1])
    #     x_label2.append(step_c)
    if err3[-1] < err3[0]:
        err_list3.append(err3[-1])
        x_label3.append(step_c)

    # if np.isnan(err1[-1]) or np.isinf(err1[-1]) or err1[-1] >= 1e5:
    #     # if np.isnan(err2[-1]) or np.isinf(err2[-1]) or err2[-1] >= 1e5:
    #     break
    # if min(err_list1) == err1[-1]:
    #     min_step1 = step_c
    #     min_err1 = err1[-1]

    # if min(err_list2) == err2[-1]:
    #     min_step2 = step_c
    #     min_err2 = err2[-1]

    if min(err_list3) == err3[-1]:
        min_step3 = step_c
        min_err3 = err3[-1]

# print("running final step size...")
# # err = NEW_ASYSONATA(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step, opt=opt)
# err1 = SemiFL_GT(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=A_L, step_c=min_step1, opt=opt)
# err2 = SemiFL_ScatterGT(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=A_L, step_c=min_step2, opt=opt)
# err3 = SemiFL(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=A_L, step_c=min_step3, opt=opt)

plt.xscale('log')
plt.yscale('log')
plt.plot(x_label1, err_list1)
plt.plot(x_label2, err_list2)
plt.plot(x_label3, err_list3)

plt.legend([r'$T = %d$' % opt.p_inv + ', min step = %.2E (err = %.2E) ALG 3' % (min_step1, min_err1),
            r'$T = %d$' % opt.p_inv + ', min step = %.2E (err = %.2E) ALG 2' % (min_step2, min_err2),
            r'$T = %d$' % opt.p_inv + ', min step = %.2E (err = %.2E) w/o GT' % (min_step3, min_err3)])
# plt.legend([ r'$\gamma = \frac{1}{3L}$', r'$\gamma = \frac{1}{10L}$', r'$\gamma = \frac{1}{30L}$'])

plt.ylabel("$ ||x - x_{opt}||_2$")
plt.xlabel("step size")
plt.title("(%d rounds)(n_c: %d) number of agents: %d" % (opt.c_rounds, opt.n_cluster, n_agents))
plt.grid()
# save_img(plt, "SEMIFL_LR_search_%.2f" % opt.omega, opt)

plt.clf()
plt.ylim([1e-8, 10])
plt.yscale('log')
plt.plot(err1)
# plt.plot(err2)
plt.plot(err3)

plt.legend([r'$\gamma = %.2E$' % min_step1 + ', test1',
            # r'$\gamma = %.2E$' % min_step2 + ', ALG 3',
            r'$\gamma = %.2E$' % min_step3 + ', test2'])
plt.ylabel(r"$||x - x_{opt}||_2$")
plt.xlabel("Communication Rounds")
plt.title("(T = %d)(Kappa: %.2f)(n_c: %d)(c_m: %d) number of agents: %d" % (opt.p_inv, kappa, opt.n_cluster, opt.sample_num, n_agents))
plt.grid()
save_img(plt, "test%.2f" % opt.omega, opt)
    # save_img(plt, "test_new_alg_%.2f" % opt.omega, opt)

pdb.set_trace()
