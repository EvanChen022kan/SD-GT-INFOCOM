from libsvm import svmutil
import numpy as np
import torch
from functions import ProxSkip, NewProx, save_img, GapMeasure, GetOpt, get_fstar, update_fstar, del_fstar, get_L, get_mse_fstar, get_chain_graph, get_circle_graph
# from functions.SONATA import ASYSONATA, OLD_ASYSONATA, NEW_ASYSONATA, ASY_GT, ASY_SGD
from functions.SemiFL import SemiFL_GT, SemiFL, SemiFL_GT4, SCAFFOLD
from functions.graph_gen import get_FC_graph, get_ring_graph, get_grid_graph, get_2star_graph, get_I_graph, get_geo_graph
from functions.RandomSkip import RandomSkip, Skip2
from functions.utils import save_plt
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

params = {'legend.fontsize': 'x-large',
          #   'figure.figsize': (15, 5),
          'lines.linewidth': 2.5,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium'
          }
plt.rcParams.update(params)

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

    AAt = input.transpose(0,1) @ input
    t0 = torch.real(torch.linalg.eig(AAt)[0])
    kappa = torch.max(t0)/torch.min(t0)
    A_L = torch.max(t0)
    print("(omega = %.2f)Kappa value: %.2f" % (omega, kappa))

n_cluster = opt.n_cluster
R = []
C = []
N_in = []
N_out = []

if opt.random:
    for i in range(n_cluster):
        # r, c, n_in, n_out = get_geo_graph(n_agents//n_cluster)
        rand_num = random.randint(0, 3)
        if rand_num == 3:
            r, c, n_in, n_out = get_FC_graph(n_agents//n_cluster)
        elif rand_num == 2:
            r, c, n_in, n_out = get_2star_graph(n_agents//n_cluster)
        elif rand_num == 1:
            r, c, n_in, n_out = get_grid_graph(n_agents//n_cluster)
        else:
            r, c, n_in, n_out = get_ring_graph(n_agents//n_cluster)
        R.append(r)
        C.append(c)
        N_in.append(n_in)
        N_out.append(n_out)
else:
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
    input_list = list(torch.split(input, n, dim = 0))
    label_list = list(torch.split(labels, n, dim=0))
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)

    mu = 0

    ''' get opt '''
    I = torch.eye(dim, dim).to(gpu)
    x_opt = torch.linalg.inv(input.transpose(0, 1) @ input ) @ input.transpose(0, 1) @ labels


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

step_c_list = [opt.lr]


skip_list = [np.random.binomial(1, prob_list[i], int(opt.c_rounds)) for i in range(n_agents)]
# delta_list = [1, 1e-3]
# delta_list = [ 1e-3]

if opt.control:
    energy_list = [random.randint(1, 100) for i in range(opt.n_cluster)]
    legend_list = []
    delta_list = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    for delta in delta_list:
        opt.delta = delta
        energy_list = [60 for i in range(opt.n_cluster)]
        lam2 = 1e-1
        lam1 = 1e-1
        K_val = []
        nc_agents = n_agents//opt.n_cluster
        sample_list = []
        sample_val_list = []
        for K in range(1, 100):
            # for i in range(opt.n_cluster):
            val_list = []
            for j in range(nc_agents):
                beta = (nc_agents - (j+1))/nc_agents
                val_list.append(1/(1 - beta**2)**4 + (1/K)**(1/2) + (1/(K*(1 - beta**2)**2))**(2/3) + lam1*(1 - beta)*sum(energy_list) + lam2*K*opt.delta*sum(energy_list))
            # print(val_list)
            # pdb.set_trace()
            sample_list.append(np.argmin(val_list)+1)
            sample_val_list.append(np.min(val_list))
            # K_val.append((1/K)**(1/2) + lam2*K*opt.delta*sum(energy_list))
        
        print(np.argmin(sample_val_list)+1)
        K_opt = np.argmin(sample_val_list)+1
        print(sample_list[np.argmin(sample_val_list)])
        sample_num_opt = sample_list[np.argmin(sample_val_list)]
        # pdb.set_trace()
        # pdb.set_trace()

        err1, errz, ener1 = SemiFL_GT4(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=K_opt, sample_num_opt=sample_num_opt,
                                    n_agents=n_agents, L=A_L, step_c=opt.lr, control=1, energy_list=energy_list, opt=opt)
        err2, errz, ener2 = SemiFL_GT4(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=1, n_agents=n_agents, L=A_L, step_c=opt.lr, control=2, energy_list=energy_list, opt=opt)
        plt.plot(ener2, err2[1:], '--')
        plt.plot(ener1, err1[1:])
        legend_list.append('SDGT %.2E' % opt.delta)
        legend_list.append('SDGT control %.2E' % opt.delta)



    plt.yscale('log')
    plt.xlim([0, 20000])
    plt.ylim([1e-7, 10])

    # plt.legend(['SD-GT + control',' SD-GT'])
    plt.ylabel(r"$||x - x_{opt}||_2$")
    plt.xlabel("Total Energy Cost")
    plt.title('Synthetic Data', loc='right')
    plt.grid()
    save_img(plt, "CONTROL_noleg_%d_%d" % (opt.sample_num, opt.p_inv), opt)
    plt.legend(legend_list)
    save_img(plt, "CONTROL_%d_%d" % (opt.sample_num, opt.p_inv), opt)


else:
    sample_num_list = [1, 3, 5]
    # sample_num_list = [1, 5]
    SCAFF_list = []
    Fed_list = []
    SDGT_list = []
    legend_list = []
    for i in range(len(sample_num_list)):
        opt.sample_num = sample_num_list[i]
        # print("Progress: %d/%d, step_c = %.2E" % ((i+1), len(step_c_list), step_c))
        print('sample_num: %d' % opt.sample_num)

        err1, errz1 = SCAFFOLD(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=A_L, step_c=opt.lr, opt=opt)
        err2, errz = SemiFL_GT4(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=int(opt.p_inv), n_agents=n_agents, L=A_L, step_c=opt.lr, opt=opt)
        err3 = SemiFL(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=A_L, step_c=opt.lr, opt=opt)

        plt.plot(err3, '--')
        plt.plot(err1, ':')
        plt.plot(err2, '-')
        legend_list.append('FedAvg, samp = %d' % opt.sample_num)
        legend_list.append('SCAFFOLD, samp = %d' % opt.sample_num)
        legend_list.append('SD-GT, samp = %d' % opt.sample_num)
        SCAFF_list.append(err1)
        SDGT_list.append(err2)
        Fed_list.append(err3)



    '''plt setting'''
    # plt.legend(['FedAvg', 'SD-GT'])
    plt.ylim([1e-12, 10])
    plt.yscale('log')
    plt.title('Synthetic Data', loc='right')
    plt.ylabel(r"$||x - x_{opt}||_2$")
    plt.xlabel("Global Aggregation Rounds")
    plt.grid()
    save_img(plt, "SYN_compRAND_noleg_%df" % ( opt.p_inv), opt)
    plt.legend(legend_list)
    save_img(plt, "SYN_compRAND_%df" % (opt.p_inv), opt)

    save_plt(SCAFF_list, "syn_SCAFFOLD_%s_%s_%.2E_acc_%d_%df" % (opt.model, opt.dataset, opt.lr, opt.sample_num, opt.p_inv), opt)
    save_plt(Fed_list, "syn_FedAvg_%s_%s_%.2E_acc_%d_%df" % (opt.model, opt.dataset, opt.lr, opt.sample_num, opt.p_inv), opt)
    save_plt(SDGT_list, "syn_SDGT_%s_%s_%.2E_acc_%d_%df" % (opt.model, opt.dataset, opt.lr, opt.sample_num, opt.p_inv), opt)



    plt.clf()
    plt.ylim([1e-12, 10])
    plt.yscale('log')
    # plt.plot(merr1)
    plt.plot(err2)
    plt.plot(err3)

    plt.legend([
                # r'$\gamma = %.2E$' % min_step1 + ', GT', 
                r'$\gamma = %.2E$' % min_step2 + ', ALG test', 
                r'$\gamma = %.2E$' % min_step3 + ', w/o gt'])
    plt.ylabel(r"$||x - x_{opt}||_2$")
    plt.xlabel("Communication Rounds")
    plt.title("(T = %d)(Kappa: %.2f)(n_c: %d)(c_m: %d)(beta:%.2f) number of agents: %d" % (opt.p_inv, kappa, opt.n_cluster, opt.sample_num, opt.beta, n_agents))
    plt.grid()
    if opt.use_grid:
        save_img(plt, "ALG_compGRID_%.2f" % opt.omega, opt)
    elif opt.use_ring:
        save_img(plt, "ALG_compRING_%.2f" % opt.omega, opt)
    elif opt.use_2star:
        save_img(plt, "ALG_comp2STAR_%.2f" % opt.omega, opt)
    else:
        save_img(plt, "ALG_compFC_%.2f" % opt.omega, opt)
        # save_img(plt, "test_new_alg_%.2f" % opt.omega, opt)

    pdb.set_trace()
