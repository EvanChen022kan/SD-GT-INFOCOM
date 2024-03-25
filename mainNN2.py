from libsvm import svmutil
import numpy as np
import torch
from functions import save_img
from functions.utils import save_plt
from functions.SemiFL_NN import SemiFL_NN, SemiFLGT_NN, SemiFLGT2_NN, SCAFFOLD_NN

from functions.graph_gen import get_FC_graph, get_ring_graph, get_grid_graph, get_2star_graph, get_geo_graph
import matplotlib.pyplot as plt
import matplotlib
from settings import Option

import scipy.io as sio
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cvxpy as cp
import pdb

matplotlib.use('Agg')
torch.manual_seed(123456)
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

    t_input.append(mnist_test[i][0])
    t_labels.append(torch.tensor(mnist_test[i][1]))

t_input = torch.stack(t_input, dim=0).to(gpu)
t_labels = torch.stack(t_labels, dim=0).to(gpu)



# pdb.set_trace()
    # mu = 0

''' get opt '''

# pdb.set_trace()

Fedplt_list = []
SDGTplt_list = []
SCAFFplt_list = []
# if opt.dataset == 'MNIST':
#     plt.ylim([0.6, 1])
#     plt.yticks((0.6, 0.7,0.8,0.9,1))
LR = opt.lr
# K_list = [10, 3, 1]
K_list = [3]

config_list = [(12,6), (24,6), (36,6), (46,6), (30,2), (30,5), (30,2)]
'setting1'
if opt.setting1:
    config_list = [(40,2), (40,10), (40,20)]
'setting2'
if opt.setting2:
    config_list = [(48,6), (36,6), (24,6), (12,6)]

list1 = []
list2 = []
list3 = []
for config in config_list:
    print('running_config: ', config)
    n_agents = config[0]
    opt.n_cluster = config[1]

    n_cluster = opt.n_cluster
    R = []
    C = []
    N_in = []
    N_out = []
    if opt.random:
        for i in range(n_cluster):
            r, c, n_in, n_out = get_geo_graph(n_agents//n_cluster)
            R.append(r)
            C.append(c)
            N_in.append(n_in)
            N_out.append(n_out)
    for K in K_list:
        print('D2D communication: %d' % K)
        nc_agents = n_agents//n_cluster
        opt.sample_num = int(0.5*nc_agents)
        it2, err2, acc2 = SemiFLGT2_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=K, n_agents=n_agents, opt=opt)
        it3, err3, acc3 = SCAFFOLD_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=K, n_agents=n_agents, opt=opt)
        SCAFFplt_list.append((it3, acc3))

        it1, err1, acc1 = SemiFL_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, LR, T=K, n_agents=n_agents, opt=opt)

        plt.plot(it1, acc1, '--', color='blue')
        plt.plot(it3, acc3, ':', color='orange')
        plt.plot(it2, acc2, '-', color='green')

        Fedplt_list.append((it1, acc1))
        SDGTplt_list.append((it2, acc2))
        legend_list = []
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
        save_img(plt, "%s_%s_%s_%.2E_%d_%d_%d_%df" % (opt.name, opt.model, opt.dataset, opt.lr, n_agents, opt.n_cluster, opt.sample_num, opt.p_inv), opt)
        plt.clf()   
        list1.append(SCAFFplt_list)
        list2.append(Fedplt_list)
        list3.append(SDGTplt_list)

        save_plt([config_list, list1], "%s_%d_%d_SCAFFOLD" % (opt.dataset, config[0], config[1]), opt)
        save_plt([config_list, list2], "%s_%d_%d_FedAvg" % (opt.dataset, config[0], config[1]), opt)
        save_plt([config_list, list3], "%s_%d_%d_SDGT" % (opt.dataset, config[0], config[1]), opt)

save_plt([config_list, list1], "%s_SCAFFOLD_%s_%s_%.2E_%d_%d_%d_%df" % (opt.name,opt.model, opt.dataset, opt.lr, n_agents, opt.n_cluster,opt.sample_num, opt.p_inv), opt)
save_plt([config_list, list2], "%s_FedAvg_%s_%s_%.2E_%d_%d_%d_%df" % (opt.name,opt.model, opt.dataset, opt.lr, n_agents, opt.n_cluster,opt.sample_num, opt.p_inv), opt)
save_plt([config_list, list3], "%s_SDGT_%s_%s_%.2E_%d_%d_%d_%df" % (opt.name,opt.model, opt.dataset, opt.lr, n_agents, opt.n_cluster,opt.sample_num, opt.p_inv), opt)
pdb.set_trace()
