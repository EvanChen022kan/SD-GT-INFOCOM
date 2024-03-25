from libsvm import svmutil
import numpy as np
import torch
from functions import ProxSkip, NewProx, save_img, GapMeasure, GetOpt, get_fstar, update_fstar, del_fstar, get_L, get_mse_fstar, get_chain_graph, get_circle_graph
from functions.SONATA import ASYSONATA, OLD_ASYSONATA, NEW_ASYSONATA, ASY_GT, ASY_SGD
from functions.RandomSkip import RandomSkip
from functions.sync import SYNC_RUNMASS, OLD_SYNC_RUNMASS, NEW_SYNC_RUNMASS
from functions.central import CEN_RUNMASS, CEN_RUNMASS2, CEN_RUNMASS3, CEN_RUNMASS4
from functions.sync_alg import SYNC_ALG_1, SYNC_ALG_2, SYNC_ALG_3
from functions.gradfn import msefn, msegfn
from scipy.optimize import minimize, fsolve
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
if opt.data_source == 'w8a':
    N_per_agent = 200
    labels, sparse_x = svmutil.svm_read_problem('data/w8a.txt')
    for i in range(len(sparse_x)):
        x = np.zeros(dim)
        if len(sparse_x[i]):
            x[np.array(list(sparse_x[i].keys())) - 1] = np.array(list(sparse_x[i].values()))
        sparse_x[i] = x

    input = torch.from_numpy(np.stack(sparse_x)).to(gpu)
    labels = torch.from_numpy(np.stack(labels)).to(gpu)
else:
    opt.dim = dim = 200
    N_per_agent = 30

    # omega = torch.tensor(0.1)
    # x0 = torch.randn(dim, dtype=torch.double).to(gpu)
    # N = N_per_agent*n_agents
    # input = torch.zeros((N, dim), dtype=torch.double)
    # input[:, 0] = torch.randn(N, dtype=torch.double)/torch.sqrt(1 - omega**2)
    # for i in range(1, dim):
    #     input[:, i] = omega*input[:, i-1] + torch.randn(N, dtype=torch.double)
    # input = input.to(gpu)
    # noise = 0.04*torch.randn(N, dtype=torch.double).to(gpu)
    # labels = input @ x0 + noise

    # pdb.set_trace()
    omega = opt.omega
    input = torch.randn((N_per_agent*n_agents, dim), dtype=torch.double).to(gpu)
    x_init = torch.randn((dim), dtype=torch.double).to(gpu)
    # for i in range(n_agents):
    #     input[i*N_per_agent: (i+1)*N_per_agent] = input[i*N_per_agent: (i+1)*N_per_agent]/torch.norm(input[i*N_per_agent: (i+1)*N_per_agent])
    for i in range(input.shape[1]):
        if i == 0:
            input[:, i] = input[:, i]/np.sqrt((1 - omega**2))
        else:
            input[:, i] = omega*input[:, i-1] + input[:, i]

    noise = torch.randn((N_per_agent*n_agents), dtype=torch.double).to(gpu)
    labels = input @ x_init + 0.2*noise

    # mat = sio.loadmat('ab.mat')
    # input = torch.from_numpy(mat['A_Gen']).type(torch.double).to(gpu)
    # labels = torch.from_numpy(mat['b_Gen']).type(torch.double).to(gpu)

    AAt = input.transpose(0,1) @ input
    t0 = torch.real(torch.linalg.eig(AAt)[0])
    kappa = torch.max(t0)/torch.min(t0)
    A_L = torch.max(t0)
    print("(omega = %.2f)Kappa value: %.2f" % (omega, kappa))
    # pdb.set_trace()



# nabla2_f = torch.zeros(dim, dim).to(gpu)
# for i in range(len(input)):
#     nabla2_f = nabla2_f + 1/4*torch.einsum('n,k->nk', input[i], input[i])
# nabla2_f /= len(input)
# L = torch.norm(nabla2_f)
# G, N_in = get_chain_graph(n_agents, batch_size=5)


R, C, N_in, N_out = get_circle_graph(n_agents, N_out_num=3)
mat = sio.loadmat('RC.mat')
R = mat['R']
C = mat['C']
N_in = [[] for i in range(n_agents)]
N_out = [[] for i in range(n_agents)]
for i in range(n_agents):
    for j in range(n_agents):
        if R[i, j] != 0 and i != j:
            N_in[i].append(j)
        if C[j, i] != 0 and i != j:
            N_out[i].append(j)



L = get_L(input, n_agents, opt)
mu = opt.kappa*L
# pdb.set_trace()

'''get fstar'''
if opt.loss != "mse":
    x = torch.randn(dim, dtype=torch.double).to(gpu)
    # f_star, x_opt = get_fstar(n_agents)
    # x = x_opt.to(gpu)
    # f_star = GetOpt(x, input, labels, lam=mu, L=L, opt=opt)


    # dict = {4: 0.1872732951272, 10: 0.1872785084491}
    # update_fstar(dict[4], torch.randn(dim, dtype=torch.double), 4)
    # update_fstar(dict[10], torch.randn(dim, dtype=torch.double), 10)
    # del_fstar(n_agents)
    f_star, x_opt = get_fstar(n_agents)
    if x_opt is None:
        x = torch.randn(dim, dtype=torch.double).to(gpu)
        f_star = GetOpt(x, input, labels, lam=mu, L=L, opt=opt)
    x_opt = x_opt.to(gpu)

elif opt.loss == 'mse':
    N = input.shape[0]
    n = N//n_agents
    input_list = list(torch.split(input, n, dim = 0))
    label_list = list(torch.split(labels, n, dim=0))
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)
    # mu = L*1e-3
    mu = 0
    # f_star, x_opt = get_mse_fstar(input_list, label_list, mu)
    # f_star, x_opt = get_mse_fstar([input], [labels], 0)
    ''' get opt '''
    I = torch.eye(dim, dim).to(gpu)
    x_opt = torch.linalg.inv(input.transpose(0, 1) @ input ) @ input.transpose(0, 1) @ labels

    # mat = sio.loadmat('x_opt.mat')['X_opt']
    # x_opt = torch.from_numpy(mat).type(torch.double).to(gpu)
    
    # x_opt.unsqueeze(1)

    f_star = 0
f_star -= 2e-17
# x = x0 + 0.25*torch.randn(dim, dtype=torch.double).to(gpu)
x = torch.randn(dim, dtype=torch.double).to(gpu)
# f_star = GetOpt(x, input, labels, lam=mu, L=L, opt=opt)





if opt.test_prox:
    ''' Run ProxSkip'''
    # err1 = ProxSkip(x, input, labels, f_star, lam=mu, p=1/10, n_agents=n_agents, L=L, opt=opt)
    # err2 = ProxSkip(x, input, labels, f_star, lam=mu, p=1/20, n_agents=n_agents, L=L, opt=opt)
    err3 = ProxSkip(x, input, labels, f_star, lam=mu, p=1/300, n_agents=n_agents, L=L, opt=opt)
    err4 = ProxSkip(x, input, labels, f_star, lam=mu, p=1/500, n_agents=n_agents, L=L, opt=opt)
    err5 = ProxSkip(x, input, labels, f_star, lam=mu, p=1/1000, n_agents=n_agents, L=L, opt=opt)
    err6 = ProxSkip(x, input, labels, f_star, lam=mu, p=1/2000, n_agents=n_agents, L=L, opt=opt)


    # plt.style.use('ieee')
    # plt.plot(err1, label = '1/p = 10')
    # plt.plot(err2, label = '1/p = 20')
    plt.plot(err3, label = '1/p = 300')
    plt.plot(err4, label = '1/p = 500')
    plt.plot(err5, label = '1/p = 1000')
    plt.plot(err6, label='1/p = 2000')
    plt.legend()
    plt.ylabel("$f(x) - f^*$")
    plt.xlabel("Communication Rounds")
    plt.title("number of agents: %d" % n_agents)
    plt.grid()
    save_img(plt, "Proxskip", opt)

elif opt.test_ours:
    '''Run NewProx'''
    c = 1
    beta = opt.beta
    # err1 = NewProx(x, input, labels, f_star, lam=mu, p=1/100, n_agents=n_agents, L=L, LR_c = c, opt=opt)
    err2 = NewProx(x, input, labels, f_star, lam=mu, p=1/300, n_agents=n_agents, L=L, LR_c = c, beta = beta, opt=opt)
    err3 = NewProx(x, input, labels, f_star, lam=mu, p=1/500, n_agents=n_agents, L=L, LR_c = c, beta = beta, opt=opt)
    err4 = NewProx(x, input, labels, f_star, lam=mu, p=1/1000, n_agents=n_agents, L=L, LR_c = c,beta = beta,  opt=opt)
    err5 = NewProx(x, input, labels, f_star, lam=mu, p=1/2000, n_agents=n_agents, L=L,LR_c = c, beta = beta,  opt=opt)
    # err6 = NewProx(x, input, labels, f_star, lam=mu, p=1/200, n_agents=n_agents, L=L,LR_c = c,  opt=opt)


    # plt.style.use('science')
    # plt.plot(err1, label='1/p = 100')
    plt.plot(err2, label='1/p = 300, beta = %d' % beta)
    plt.plot(err3, label='1/p = 500, beta = %d' % beta)
    plt.plot(err4, label='1/p = 1000, beta = %d' % beta)
    plt.plot(err5, label='1/p = 2000, beta = %d' % beta)

    # plt.plot(err5, label='1/p = 100')
    # plt.plot(err6, label='1/p = 200')

    plt.legend()
    plt.ylabel("$f(x) - f^*$")  
    plt.xlabel("Communication Rounds")
    plt.title("number of agents: %d" % n_agents)
    plt.grid()
    save_img(plt, "Ours", opt)


# '''Comparison'''
# x = torch.randn(dim, dtype=torch.double).to(gpu)
# err2 = NewProx(x, input, labels, f_star, lam=mu, p=1/5000, n_agents=n_agents, L=L, opt=opt)
# err1 = ProxSkip(x, input, labels, f_star, lam=mu, p=1/5000, n_agents=n_agents, L=L, opt=opt)
# plt.style.use('science')
# plt.plot(err1, label='1/p = 5000, Prox')
# plt.plot(err2, label='1/p = 5000, NewProx')
# plt.legend()
# plt.ylabel("$f(x) - f^*$")
# plt.xlabel("Communication Rounds")
# plt.title("number of agents: %d" % n_agents)
# plt.grid()
# save_img(plt, "Comparison", opt)

elif opt.comp_prox:

    step_c1 = 3.5*1e-3
    step_c2 = 1.75*1e-3
    err = CEN_RUNMASS3(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c1, opt=opt)
    err1 = ProxSkip(x, input, labels, f_star, x_opt=x_opt, lam=mu, p=1/opt.p_inv, n_agents=n_agents, L=L, step_c=step_c2, opt=opt)

    plt.plot(err)
    plt.plot(err1)

    plt.legend([r'$T = %d$' % opt.p_inv + ', Ours', r'$T = %d$' % opt.p_inv + ', Proxskip'])
    plt.ylabel(r"$||x - x_{opt}||_2$")
    plt.xlabel("Communication Rounds")
    plt.title("(T = %d) number of agents: %d" % (opt.p_inv, n_agents))
    plt.grid()
    save_img(plt, "prox_comp", opt)

    pdb.set_trace()
elif opt.test_sync or opt.test_cen:
    err_list = []
    prox_err_list = []
    err_list3 = []
    err_list4 = []

    x_label = []
    x_label2 = []
    x_label3 = []
    x_label4 = []
    min_step = 3.5
    min_err = 10
    min_step2 = 10
    min_err2 = 10
    min_step3= 3.5
    min_err3 = 10
    min_step4 = 10
    min_err4 = 10
    N = 10
    round_i = 1


    start_val = (1/A_L).item()
    # step_c_list = [((i+1)/N)*0.035 for i in range(N-1)] + [((i+1)/N)*0.35 for i in range(N-1)] + [((i+1)/N)*3.5 for i in range(N-1)] + [((i+1)/N)*35 for i in range(N-1)]
    step_c_list = [((i+1)/N)*start_val for i in range(N-1)] + [((i+1)/N)*10
                                                                *start_val for i in range(N-1)] + [((i+1)/N)*100
                                                                *start_val for i in range(N-1)] + [((i+1)/N)*1000
                                                                *start_val for i in range(N-1)] + [((i+1)/N)*1e4*start_val for i in range(N-1)]
    # step_c_list = [((i+1)/N)*0.035 for i in range(N-1)] + [((i+1)/N) * 0.35 for i in range(N-1)] + [((i+1)/N)
    #                                                                                                 * 3.5 for i in range(N-1)] + [((i+1)/N)*35 for i in range(N-1)] + [((i+1)/N)*350 for i in range(N-1)]

    step_c_list[0] = 3*1e-4


    for i in range(len(step_c_list)):
        step_c = step_c_list[i]
        print("Progress: %d/%d, step_c = %.2E" % ((i+1), len(step_c_list), step_c))
        if opt.test_cen:
            # err = CEN_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
            # err = CEN_RUNMASS3(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
            err = CEN_RUNMASS3(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
            err1 = ProxSkip(x, input, labels, f_star, x_opt, opt.c_rounds, lam=mu, p=1/opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)

        else:

            err, d1 = NEW_SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
            plt.plot(d1)
            plt.ylabel("$C_r$")
            plt.xlabel("communication rounds")
            plt.legend([r'$T = %d$' % opt.p_inv + ', Step Size: %.2E' % (step_c_list[0])])
            plt.title("Change of the sequence Cr" )
            plt.grid()
            save_img(plt, "c_r", opt)
            pdb.set_trace()
            
            err1 = SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
            err2 = SYNC_ALG_1(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
            # err3 = SYNC_ALG_2(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
            err3 = SYNC_ALG_3(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)

            # err1 = OLD_SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)


        if err[-1] < err[0]:
            err_list.append(err[-1])
            x_label.append(step_c)
        if err1[-1] < err1[0]:
            prox_err_list.append(err1[-1])
            x_label2.append(step_c)
        if opt.test_sync:
            if err2[-1] < err2[0]:
                err_list3.append(err2[-1])
                x_label3.append(step_c) 
            if err3[-1] < err3[0]:
                err_list4.append(err3[-1])
                x_label4.append(step_c)
            if min(err_list3) == err2[-1]:
                min_step3 = step_c
                min_err3 = err2[-1]
            if min(err_list4) == err3[-1]:
                min_step4 = step_c
                min_err4 = err3[-1]

        if np.isnan(err[-1]) or np.isinf(err[-1]) or err[-1] >= 1e1:
            if np.isnan(err1[-1]) or np.isinf(err1[-1]) or err1[-1] >= 1e1:
                if opt.test_sync:
                    if np.isnan(err2[-1]) or np.isinf(err2[-1]) or err2[-1] >= 1e1:
                        if np.isnan(err3[-1]) or np.isinf(err3[-1]) or err3[-1] >= 1e1:
                            break
                else:
                    break
        if min(err_list) == err[-1]:
            min_step = step_c
            min_err = err[-1]
            # best_err_plot = err
        if min(prox_err_list) == err1[-1]:
            min_step2 = step_c
            min_err2 = err1[-1]
            # best_err_plot1 = err1

    print("running final step size...")
    if opt.test_cen:
        # err = CEN_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
        # err = CEN_RUNMASS3(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds*5, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step, opt=opt)
        err = CEN_RUNMASS3(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step, opt=opt)
        err1 = ProxSkip(x, input, labels, f_star, x_opt, opt.c_rounds, lam=mu, p=1/opt.p_inv, n_agents=n_agents, L=L, step_c=min_step2, opt=opt)
        
    else:
        err = NEW_SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, round_i*opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step, opt=opt)
        err1 = SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, round_i*opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step2, opt=opt)
        err2 = SYNC_ALG_1(x, input, labels, R, C, N_in, N_out, x_opt, f_star, round_i*opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step3, opt=opt)
        # err3 = SYNC_ALG_2(x, input, labels, R, C, N_in, N_out, x_opt, f_star, round_i*opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step4, opt=opt)
        err3 = SYNC_ALG_3(x, input, labels, R, C, N_in, N_out, x_opt, f_star, round_i*opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step4, opt=opt)

        # err1 = OLD_SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step2, opt=opt)
    # plt.plot(err5)
    # plt.plot(err6)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x_label, err_list)
    plt.plot(x_label2, prox_err_list)
    if opt.test_sync:
        plt.plot(x_label3, err_list3)
        plt.plot(x_label4, err_list4)

    # plt.plot(err2)
    # plt.plot(err3)
    # plt.plot(err4)
    if opt.test_cen:
        plt.legend([r'$T = %d$' % opt.p_inv + ', (Ours(ideal)) min step = %.2E (err = %.2E)' % (min_step, min_err),
                    r'$T = %d$' % opt.p_inv + ', (Proxskip) min step = %.2E (err = %.2E)' % (min_step2, min_err2)])
    else:
        plt.legend([r'$T = %d$' % opt.p_inv + ', (Alg2) min step = %.2E (err = %.2E)' % (min_step, min_err),
                    r'$T = %d$' % opt.p_inv + ', (Alg1) min step = %.2E (err = %.2E)' % (min_step2, min_err2),
                    r'$T = %d$' % opt.p_inv + ', (K-GT) min step = %.2E (err = %.2E)' % (min_step3, min_err3),
                    r'$T = %d$' % opt.p_inv + ', (LSGT) min step = %.2E (err = %.2E)' % (min_step4, min_err4)
                    ])
    plt.ylabel("$ ||x - x_{opt}||_2$")
    plt.xlabel("step size")
    if opt.test_sync:
        plt.title("(%d rounds)(Synchronous) number of agents: %d" % (opt.c_rounds, n_agents))
        
        plt.grid()
        save_img(plt, "All_sync_%.2f" % opt.omega, opt)

        plt.clf()
        plt.ylim([1e-6, 10])
        plt.yscale('log')
        plt.plot(err)
        plt.plot(err1)
        plt.plot(err2)
        plt.plot(err3)

        plt.legend([r'$\gamma = %.2E$' % min_step + ', Alg2', r'$\gamma = %.2E$' % min_step2 + ', Alg1',
                    r'$\gamma = %.2E$' % min_step3 + ', K-GT'
                    , r'$\gamma = %.2E$' % min_step4 + ', LSGT'
                    ])
        plt.ylabel(r"$||x - x_{opt}||_2$")
        plt.xlabel("Communication Rounds")
        plt.title("(T = %d)(Kappa: %.2f) number of agents: %d" % (opt.p_inv, kappa, n_agents))
        plt.grid()
        save_img(plt, "All_sync_comp_%.2f" % opt.omega, opt)
    else:
        plt.title("(%d rounds)(centralized) number of agents: %d" % (opt.c_rounds, n_agents))
        plt.grid()
        save_img(plt, "old_cen_mse_%.2f" % opt.omega, opt)

        plt.clf()
        plt.ylim([1e-6, 10])
        plt.yscale('log')
        plt.plot(err)
        plt.plot(err1)



        plt.legend([r'$\gamma = %.2E$' % min_step + ', Ours', r'$\gamma = %.2E$' % min_step2 + ', Proxskip'])
        plt.ylabel(r"$||x - x_{opt}||_2$")
        plt.xlabel("Communication Rounds")
        plt.title("(T = %d)(Kappa: %.2f) number of agents: %d" % (opt.p_inv, kappa, n_agents))
        plt.grid()
        save_img(plt, "old_comp_%.2f" % opt.omega, opt)

    pdb.set_trace()

elif opt.test_asy:

    err_list = []
    err_list2 = []
    err_list3 = []

    x_label = []
    x_label2 = []
    x_label3 = []

    min_step = 3.5
    min_err = 10
    min_step2 = 3.5
    min_err2 = 10
    min_step3 = 3.5
    min_err3 = 10
    N = 10

    start_val = (1/A_L).item()
    # step_c_list = [((i+1)/N)*0.035 for i in range(N-1)] + [((i+1)/N)*0.35 for i in range(N-1)] + [((i+1)/N)*3.5 for i in range(N-1)] + [((i+1)/N)*35 for i in range(N-1)]
    step_c_list = [((i+1)/N)*start_val for i in range(N-1)] + [((i+1)/N)*10
                                                                *start_val for i in range(N-1)] + [((i+1)/N)*100
                                                                *start_val for i in range(N-1)] + [((i+1)/N)*1000
                                                                *start_val for i in range(N-1)] + [((i+1)/N)*1e4*start_val for i in range(N-1)]

    
    prob_list = [1/opt.p_inv for i in range(n_agents)]
    skip_list = [np.random.binomial(1, prob_list[i], int(opt.c_rounds)) for i in range(n_agents)]

    dummy_list = torch.zeros(int(opt.c_rounds))
    for i in range(len(dummy_list)):
        if i % opt.p_inv == 0:
            dummy_list[i] = 1
    skip_list = [dummy_list for i in range(n_agents)]

    for i in range(len(step_c_list)):
        step_c = step_c_list[i]
        print("Progress: %d/%d, step_c = %.2E" % ((i+1), len(step_c_list), step_c))
        # err = NEW_ASYSONATA(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
        err = RandomSkip(x, input, labels, skip_list, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
        # err = ASY_GT(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
        err1 = ASYSONATA(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
        err2 = ASY_SGD(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)

        # err1 = OLD_ASYSONATA(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)


        if err[-1] < err[0]:
            err_list.append(err[-1])
            x_label.append(step_c)
        if err1[-1] < err1[0]:
            err_list2.append(err1[-1])
            x_label2.append(step_c)
        if err2[-1] < err2[0]:
            err_list3.append(err2[-1])
            x_label3.append(step_c)
        if np.isnan(err[-1]) or np.isinf(err[-1]) or err[-1] >= 1e5:
            if np.isnan(err1[-1]) or np.isinf(err1[-1]) or err1[-1] >= 1e5:
                break
        if min(err_list) == err[-1]:
            min_step = step_c
            min_err = err[-1]
        # if opt.test_cen:
        if min(err_list2) == err1[-1]:
            min_step2 = step_c
            min_err2 = err1[-1]
        if min(err_list3) == err2[-1]:
            min_step3 = step_c
            min_err3 = err2[-1]

    print("running final step size...")
    # err = NEW_ASYSONATA(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step, opt=opt)
    err = ASY_GT(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step, opt=opt)
    err1 = ASYSONATA(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step2, opt=opt)
    err2 = ASY_SGD(x, input, labels, R, C, N_in, N_out, x_opt, f_star, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step3, opt=opt)

    # plt.plot(err5)
    # plt.plot(err6)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x_label, err_list)
    plt.plot(x_label2, err_list2)

    # plt.plot(err2)
    # plt.plot(err3)
    # plt.plot(err4)
    plt.legend([r'$T = %d$' % opt.p_inv + ', (no mass update) min step = %.2E (err = %.2E)' % (min_step, min_err),
                r'$T = %d$' % opt.p_inv + ', (ours) min step = %.2E (err = %.2E)' % (min_step2, min_err2)])
    # plt.legend([ r'$\gamma = \frac{1}{3L}$', r'$\gamma = \frac{1}{10L}$', r'$\gamma = \frac{1}{30L}$'])

    plt.ylabel("$ ||x - x_{opt}||_2$")
    plt.xlabel("step size")
    plt.title("(%d rounds) number of agents: %d" % (opt.c_rounds, n_agents))
    plt.grid()
    save_img(plt, "asy_LR_search_%.2f" % opt.omega, opt)

    plt.clf()
    plt.ylim([1e-8, 10])
    plt.yscale('log')
    plt.plot(err2)
    plt.plot(err)
    plt.plot(err1)


    plt.legend([r'$\gamma = %.2E$' % min_step + ', SGD', r'$\gamma = %.2E$' % min_step + ', GT only', r'$\gamma = %.2E$' % min_step2 + ', GT + Mass update'])
    plt.ylabel(r"$||x - x_{opt}||_2$")
    plt.xlabel("Communication Rounds")
    plt.title("(T = %d)(Kappa: %.2f) number of agents: %d" % (opt.p_inv, kappa, n_agents))
    plt.grid()
    save_img(plt, "asy_comp_%.2f" % opt.omega, opt)
    pdb.set_trace()

elif opt.sync_only:
    err_list = []
    prox_err_list = []
    err_list3 = []
    err_list4 = []

    x_label = []
    x_label2 = []

    min_step = 3.5
    min_err = 10
    min_step2 = 10
    min_err2 = 10

    N = 10

    round_i = 2

    start_val = (1/A_L*1e-1).item()
    step_c_list = [((i+1)/N)*start_val for i in range(N-1)] + [((i+1)/N)*10 * start_val for i in range(N-1)] + [((i+1)/N)
                                                                                                                * 100 * start_val for i in range(N-1)] + [((i+1)/N)*1000 * start_val for i in range(N-1)] + [((i+1)/N)*1e4*start_val for i in range(N-1)]

    for i in range(len(step_c_list)):
        step_c = step_c_list[i]
        print("Progress: %d/%d, step_c = %.2E" % ((i+1), len(step_c_list), step_c))

        err = NEW_SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)
        err1 = SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=step_c, opt=opt)

        if err[-1] < err[0]:
            err_list.append(err[-1])
            x_label.append(step_c)
        if err1[-1] < err1[0]:
            prox_err_list.append(err1[-1])
            x_label2.append(step_c)


        if np.isnan(err[-1]) or np.isinf(err[-1]) or err[-1] >= 1e1:
            if np.isnan(err1[-1]) or np.isinf(err1[-1]) or err1[-1] >= 1e1:
                break
        if min(err_list) == err[-1]:
            min_step = step_c
            min_err = err[-1]

        if min(prox_err_list) == err1[-1]:
            min_step2 = step_c
            min_err2 = err1[-1]


    print("running final step size...")

    err = NEW_SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, round_i*opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step, opt=opt)
    err1 = SYNC_RUNMASS(x, input, labels, R, C, N_in, N_out, x_opt, f_star, round_i*opt.c_rounds, lam=mu, T=opt.p_inv, n_agents=n_agents, L=L, step_c=min_step2, opt=opt)

    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x_label, err_list)
    plt.plot(x_label2, prox_err_list)

    plt.legend([r'$T = %d$' % opt.p_inv + ', (New) min step = %.2E (err = %.2E)' % (min_step, min_err),
                r'$T = %d$' % opt.p_inv + ', (Old) min step = %.2E (err = %.2E)' % (min_step2, min_err2)])
    plt.ylabel("$ ||x - x_{opt}||_2$")
    plt.xlabel("step size")
    plt.title("(%d rounds)(Synchronous) number of agents: %d" % (opt.c_rounds, n_agents))

    plt.grid()
    save_img(plt, "sync_%.2f" % opt.omega, opt)

    plt.clf()
    plt.ylim([1e-6, 10])
    plt.yscale('log')
    plt.plot(err)
    plt.plot(err1)

    plt.legend([r'$\gamma = %.2E$' % min_step + ', New', r'$\gamma = %.2E$' % min_step2 + ', Old'])
    plt.ylabel(r"$||x - x_{opt}||_2$")
    plt.xlabel("Communication Rounds")
    plt.title("(T = %d)(Kappa: %.2f) number of agents: %d" % (opt.p_inv, kappa, n_agents))
    plt.grid()
    save_img(plt, "sync_comp_%.2f" % opt.omega, opt)

