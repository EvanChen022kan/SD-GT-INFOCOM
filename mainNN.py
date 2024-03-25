from libsvm import svmutil
import numpy as np
import torch
from functions import ProxSkip, NewProx, save_img, GapMeasure, GetOpt, get_fstar, update_fstar, del_fstar, get_L, get_mse_fstar, get_chain_graph, get_circle_graph
# from functions.SONATA import ASYSONATA, OLD_ASYSONATA, NEW_ASYSONATA, ASY_GT, ASY_SGD
# from functions.SemiFL import SemiFL_GT, SemiFL, SemiFL_GT2, SemiFL_ScatterGT, SemiFL_ScatterGT2
from functions.utils import save_plt
from functions.SemiFL_NN import SemiFL_NN, SemiFLGT_NN, SemiFLGT2_NN, SCAFFOLD_NN

from functions.graph_gen import get_FC_graph, get_ring_graph, get_grid_graph, get_2star_graph, get_geo_graph
from functions.RandomSkip import RandomSkip, Skip2
import matplotlib.pyplot as plt
import matplotlib
from settings import Option
import pickle
import scienceplots
import random
import scipy.io as sio
import time
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cvxpy as cp
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

# m = 30
# n = 20
# np.random.seed(1)
# A = np.random.randn(m, n)
# b = np.random.randn(m)

# # Construct the problem.
# x1 = cp.Variable(1)
# x2 = cp.Variable(1)

# objective = cp.Minimize(1/cp.maximum(x1, x2) + (1 - x1)*3 + (1 - x2)*50)
# constraints = [0 <= x1, x1 <= 0.8, 0 <= x2, x2 <= 0.8]
# prob = cp.Problem(objective, constraints)

# result = prob.solve()
# # The optimal value for x is stored in `x.value`.
# print(x1.value)
# print(x2.value)

# print(constraints[0].dual_value)
# pdb.set_trace()


''' Load data and convert to numpy arrays'''
if opt.dataset == 'MNIST':
    mnist = datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())
elif opt.dataset == 'CIFAR10':
    mnist = datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    mnist_test = datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
elif opt.dataset == 'CIFAR100':
    mnist = datasets.CIFAR100(root='../data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.CIFAR100(root='../data', train=False, download=True, transform=transforms.ToTensor())


input = []
labels = []
for i in range(len(mnist)):
    # if opt.model == "FNN":
    #     input.append(mnist[i][0].flatten())
    # else:
    input.append(mnist[i][0])
    labels.append(torch.tensor(mnist[i][1]))
# pdb.set_trace()

'''sort all even labels'''
# input = torch.stack(input, dim = 0).to(gpu)
# labels = torch.stack(labels, dim=0).to(gpu)
# even_labels, indices = torch.sort(labels[::2])
# labels[::2] = even_labels
# input[::2] = input[::2][indices]
'''sort all labels'''
input = torch.stack(input, dim=0).to(gpu)
labels = torch.stack(labels, dim=0).to(gpu)
labels, indices = torch.sort(labels)
input = input[indices]

# pdb.set_trace()

t_input = []
t_labels = []
for i in range(len(mnist_test)):
    # if opt.model == "FNN":
    #     t_input.append(mnist_test[i][0].flatten())
    # else:
    t_input.append(mnist_test[i][0])
    t_labels.append(torch.tensor(mnist_test[i][1]))

t_input = torch.stack(t_input, dim=0).to(gpu)
t_labels = torch.stack(t_labels, dim=0).to(gpu)
# t_input = input
# t_labels = labels

n_cluster = opt.n_cluster
R = []
C = []
N_in = []
N_out = []

if opt.random:
    for i in range(n_cluster):
        r, c, n_in, n_out = get_geo_graph(n_agents//n_cluster)

        # rand_num = random.randint(0, 3)
        # if rand_num == 3:
        #     r, c, n_in, n_out = get_FC_graph(n_agents//n_cluster)
        # elif rand_num == 2:
        #     r, c, n_in, n_out = get_2star_graph(n_agents//n_cluster)
        # elif rand_num == 1:
        #     r, c, n_in, n_out = get_grid_graph(n_agents//n_cluster)
        # else:
        #     r, c, n_in, n_out = get_ring_graph(n_agents//n_cluster)
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
# if opt.loss == 'mse':
#     N = input.shape[0]
#     n = N//n_agents
#     input_list = list(torch.split(input, n, dim=0))
#     label_list = list(torch.split(labels, n, dim=0))
#     if len(input_list) != n_agents:
#         input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
#         input_list.pop(-1)
#         label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
#         label_list.pop(-1)

    # mu = 0

''' get opt '''

# pdb.set_trace()
if opt.new_control:
    # pdb.set_trace()
    delta_list = [1e-2, 1]
    energy_list = [random.randint(1, 100) for i in range(opt.n_cluster)]
    legend_list = []
    min_err = 1e10
    for delta in delta_list:
        opt.delta = delta
        print("current delta: %.2e" % opt.delta)
        '''config 1'''
        lam1 = 10
        lam2 = 1e-1
        lam3 = 1e-1
        # '''config 2'''
        # lam1 = 1e-1
        # lam2 = 1e-1
        # lam3 = 1e-1
        '''config 3'''
        lam1 = 10
        lam2 = 1
        lam3 = 1
        '''config 4'''
        lam1 = 1
        lam2 = 1e-1
        lam3 = 10

        '''config 5'''
        lam1 = 1
        lam2 = 1e-1
        lam3 = 1e-2

        '''config 6'''
        lam1 = 10
        lam2 = 1e-1
        lam3 = 1e-2

        '''config 7'''
        lam1 = 10
        lam2 = 1e-1
        lam3 = 1e-3

        '''config 8'''
        lam1 = 10
        lam2 = 1e-1
        lam3 = 1e-4

        '''config 9'''
        lam1 = 10
        lam2 = 1
        lam3 = 1e-3

        '''config 10'''
        lam1 = 10
        lam2 = 10
        lam3 = 1e-3

        '''config 11'''
        lam1 = 1
        lam2 = 1
        lam3 = 1e-5
        '''config 12'''
        lam1 = 1
        lam2 = 1e-1
        lam3 = 1e-5

        '''config 13'''
        lam1 = 1e-1
        lam2 = 1
        lam3 = 1e-5
        K = 3
        nc_agents = n_agents//opt.n_cluster
        LR = opt.lr
        it1, err1, acc1, ener1 = SemiFLGT2_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=K, n_agents=n_agents, control=3, lam = [lam1, lam2, lam3], energy_list=energy_list, opt=opt)
        it2, err2, acc2, ener2 = SemiFLGT2_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=K, n_agents=n_agents, control=2, energy_list=energy_list, opt=opt)
        if delta == delta_list[0]:
            acc = acc2[1:]
        # pdb.set_trace()
        plt.clf()
        plt.plot(ener2, acc, '--')
        plt.plot(ener1, acc1[1:])
        legend_list = []
        legend_list.append('SD-GT, delta = %.2E' % delta)
        legend_list.append('SD-GT + control, delta = %.2E' % delta)
        min_err = 1e10
        
        if ener1[-1] < min_err:
            min_err = ener1[-1]
        if ener2[-1] < min_err:
            min_err = ener2[-1]
        plt.xlim([0, min_err])
        
        plt.title('%s' % opt.dataset, loc='right')
        plt.ylabel("Test Accuracy")
        plt.xlabel("Total Energy Cost")
        plt.grid()
        save_img(plt, "%s_%s_%s_compRAND_acc_noleg_%d_%df" % (opt.name, opt.model, opt.dataset, opt.sample_num, opt.p_inv), opt)
        plt.legend(legend_list)
        save_img(plt, "%s_%s_%s_compRAND_acc_%d_%df" % (opt.name, opt.model, opt.dataset, opt.sample_num, opt.p_inv), opt)
elif opt.control:
    delta_list = [1e-4, 1e-2, 1]
    energy_list = [random.randint(1, 100) for i in range(opt.n_cluster)]
    legend_list = []
    min_err = 1e10
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
        LR = opt.lr
        # K = 10
        legend_list = []
        it1, err1, acc1, ener1 = SemiFLGT2_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=K_opt, sample_num_opt=sample_num_opt, n_agents=n_agents, control=1, opt=opt)
        it2, err2, acc2, ener2 = SemiFLGT2_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=1, n_agents=n_agents, control=2, opt=opt)
        if delta == 1e-4:
            acc = acc2[1:]
        plt.plot(ener2, acc, '--')
        plt.plot(ener1, acc1[1:])
        legend_list.append('SD-GT, delta = %.2E' % delta)
        legend_list.append('SD-GT + control, delta = %.2E' % delta)
        if ener1[-1] < min_err:
            min_err = ener1[-1]
        if ener2[-1] < min_err:
            min_err = ener2[-1]

    '''plt setting'''
    # plt.legend(['FedAvg', 'SD-GT'])
    plt.xlim([0, min_err])
    plt.title('%s' % opt.dataset, loc='right')
    plt.ylabel("Test Accuracy")
    plt.xlabel("Total Energy Cost")
    plt.grid()
    save_img(plt, "CONTROL_%s_%s_compRAND_acc_noleg_%d_%df" % (opt.model, opt.dataset, opt.sample_num, opt.p_inv), opt)
    plt.legend(legend_list)
    save_img(plt, "CONTROL_%s_%s_compRAND_acc_%d_%df" % (opt.model, opt.dataset, opt.sample_num, opt.p_inv), opt)

else:
    Fedplt_list = []
    SDGTplt_list = []
    SCAFFplt_list = []
    # if opt.dataset == 'MNIST':
    #     plt.ylim([0.6, 1])
    #     plt.yticks((0.6, 0.7,0.8,0.9,1))
    LR = opt.lr
    # K_list = [10, 3, 1]
    K_list = [30]

    # K_list = [1, 3, 10]

    # K_list = [10]
    legend_list = []
    for K in K_list:
        print('D2D communication: %d' % K)
        it2, err2, acc2 = SemiFLGT2_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=K, n_agents=n_agents, opt=opt)
        it3, err3, acc3 = SCAFFOLD_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=K, n_agents=n_agents, opt=opt)
        SCAFFplt_list.append((it3, acc3))

        it1, err1, acc1 = SemiFL_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=K, n_agents=n_agents, opt=opt)

        plt.plot(it1, acc1, '--', color='blue')
        plt.plot(it3, acc3, ':', color='orange')
        plt.plot(it2, acc2, '-', color='green')

        Fedplt_list.append((it1, acc1))
        SDGTplt_list.append((it2, acc2))

        legend_list.append('FedAvg, K = %d' % K)
        legend_list.append('SCAFFOLD, K = %d' % K)
        legend_list.append('SD-GT, K = %d' % K)



    '''plt setting'''
    # plt.legend(['FedAvg', 'SD-GT'])
    plt.title('%s' % opt.dataset, loc='right')
    plt.ylabel("Test Accuracy")
    plt.xlabel("Global Aggregation Rounds")
    plt.grid()
    # save_img(plt, "%s_%s_%.2E_compRAND_acc_noleg_%d_%d_%d_%df" % (opt.model, opt.dataset,opt.lr, n_agents, opt.n_cluster ,opt.sample_num, opt.p_inv), opt)
    plt.legend(legend_list)
    save_img(plt, "%s_%s_%.2E_compRAND_acc_%d_%d_%d_%df" % (opt.model, opt.dataset, opt.lr, n_agents, opt.n_cluster, opt.sample_num, opt.p_inv), opt)

    save_plt(SCAFFplt_list, "1-SCAFFOLD_%s_%s_%.2E_compRAND_acc_%d_%d_%d_%df" % (opt.model, opt.dataset, opt.lr, n_agents, opt.n_cluster,opt.sample_num, opt.p_inv), opt)
    save_plt(Fedplt_list, "1-FedAvg_%s_%s_%.2E_compRAND_acc_%d_%d_%d_%df" % (opt.model, opt.dataset, opt.lr, n_agents, opt.n_cluster,opt.sample_num, opt.p_inv), opt)
    save_plt(SDGTplt_list, "1-SDGT_%s_%s_%.2E_compRAND_acc_%d_%d_%d_%df" % (opt.model, opt.dataset, opt.lr, n_agents, opt.n_cluster,opt.sample_num, opt.p_inv), opt)
    pdb.set_trace()
