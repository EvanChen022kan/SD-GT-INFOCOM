import torch
import matplotlib.pyplot as plt
import matplotlib
from settings import Option
from functions import ProxSkip, NewProx, save_img

import pdb

matplotlib.use('Agg')
torch.manual_seed(123456)
# plt.yscale('log')
opt = Option("train")
# gpu = torch.device('cuda:{}'.format(opt.gpu_id))
# dim = opt.dim
# n_agents = opt.n_agents

params = {'legend.fontsize': 'x-large',
          #   'figure.figsize': (15, 5),
          'lines.linewidth': 2.5,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium'
          }
plt.rcParams.update(params)


dataset = 'MNIST'
config = 'setting1'


SCAFFOLD_name = 'results/setting.0401/setting1_SCAFFOLD_TOMCNN_CIFAR100_3.00E-03_40_20_1_1000f-03-31-08-43-41.pth'
FedAvg_name = 'results/setting.0401/setting1_FedAvg_TOMCNN_CIFAR100_3.00E-03_40_20_1_1000f-03-31-08-43-41.pth'
SDGT_name = 'results/setting.0401/setting1_SDGT_TOMCNN_CIFAR100_3.00E-03_40_20_1_1000f-03-31-08-43-41.pth'

SCAFFOLD_name2 = 'results/setting.0401/setting1-2_SCAFFOLD_TOMCNN_CIFAR100_3.00E-03_40_5_4_1000f-04-01-03-38-44.pth'
FedAvg_name2 = 'results/setting.0401/setting1-2_FedAvg_TOMCNN_CIFAR100_3.00E-03_40_5_4_1000f-04-01-03-38-44.pth'
SDGT_name2 = 'results/setting.0401/setting1-2_SDGT_TOMCNN_CIFAR100_3.00E-03_40_5_4_1000f-04-01-03-38-44.pth'


'''After 2024.02'''

dataset = 'CIFAR100'



# legend_list = []
label_list = [(40, 20), (40, 10), (40, 5), (40, 2)]

SCAF = torch.load(SCAFFOLD_name)
FedA = torch.load(FedAvg_name)
SDGT = torch.load(SDGT_name)


# for i in range(len(SCAF[0])):
#     label_list.append(SCAF[0][i])

SCAF2 = torch.load(SCAFFOLD_name2)
FedA2 = torch.load(FedAvg_name2)
SDGT2 = torch.load(SDGT_name2)


# pdb.set_trace()
# new = []
# # pdb.set_trace()
# for i in range(len(SDGT[1][0][-1][1])):
#     new.append((SDGT[1][0][-1][1][i] + SDGT2[1][0][-1][1][i])/2)
# SDGT[1][1][1] = SDGT[1][2][1] = SDGT[1][0][1] = (SDGT[1][0][1][0], new)
# torch.save(SDGT, SDGT_name)
# pdb.set_trace()

# pdb.set_trace()
K = [3]
skip = 16
skip = 8

# skip = 4
# skip = 1
if opt.setting1:
    config = 'setting1'
    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(25.6,4.8))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    # pdb.set_trace()
    axes[0].plot(FedA[1][0][-1][0][::skip], FedA[1][0][-1][1][::skip], '--', color = 'blue')
    axes[0].plot(SCAF[1][0][-1][0][::skip], SCAF[1][0][-1][1][::skip], ':',  color = 'orange')
    axes[0].plot(SDGT[1][0][-1][0][::skip], SDGT[1][0][-1][1][::skip], '-',  color = 'green')
    axes[0].set_xlabel('40 agents, 20 subnets')
    axes[0].grid()

    axes[1].plot(FedA[1][0][1][0][::skip], FedA[1][0][1][1][::skip], '--', color = 'blue')
    axes[1].plot(SCAF[1][0][1][0][::skip], SCAF[1][0][1][1][::skip], ':',  color = 'orange')
    axes[1].plot(SDGT[1][0][1][0][::skip], SDGT[1][0][1][1][::skip], '-',  color = 'green')
    axes[1].set_xlabel('40 agents, 10 subnets')

    axes[1].grid()

    axes[2].plot(FedA2[1][0][-1][0][::skip], FedA2[1][0][-1][1][::skip], '--', color = 'blue')
    axes[2].plot(SCAF2[1][0][-1][0][::skip], SCAF2[1][0][-1][1][::skip], ':',  color = 'orange')
    axes[2].plot(SDGT2[1][0][-1][0][::skip], SDGT2[1][0][-1][1][::skip], '-',  color = 'green')
    axes[2].set_xlabel('40 agents, 5 subnets')
    axes[2].grid()

    axes[3].plot(FedA[1][0][0][0][::skip], FedA[1][0][0][1][::skip], '--', color = 'blue', label = 'FedAvg')
    axes[3].plot(SCAF[1][0][0][0][::skip], SCAF[1][0][0][1][::skip], ':',  color = 'orange', label = 'SCAFFOLD')
    axes[3].plot(SDGT[1][0][0][0][::skip], SDGT[1][0][0][1][::skip], '-',  color = 'green', label = 'SD-GT')
    axes[3].set_xlabel('40 agents, 2 subnets')
    axes[3].grid()

    # pdb.set_trace()

    axes[0].set_title(dataset, loc='right')
    axes[1].set_title(dataset, loc='right')
    axes[2].set_title(dataset, loc='right')
    axes[3].set_title(dataset, loc='right')

    # plt.title(dataset)
    # plt.legend()
    plt.plot(0, 0, '--', color = 'blue', label = 'FedAvg')
    plt.plot(0, 0, ':',  color = 'orange', label = 'SCAFFOLD')
    plt.plot(0, 0, '-',  color = 'green', label = 'SD-GT')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
            ncol=3, fancybox=True)
    plt.ylabel("Test Accuracy", labelpad=10)
    plt.xlabel("Global Aggregation Rounds", labelpad=30)
    save_img(plt, "%s_%s" % (config,dataset), opt)
else:
    config = 'setting2'
    SCAFFOLD_name = 'results/setting.0401/setting2_SCAFFOLD_TOMCNN_CIFAR100_3.00E-03_12_6_1_1000f-03-29-14-00-43.pth'
    FedAvg_name = 'results/setting.0401/setting2_FedAvg_TOMCNN_CIFAR100_3.00E-03_12_6_1_1000f-03-29-14-00-43.pth'
    SDGT_name = 'results/setting.0401/setting2_SDGT_TOMCNN_CIFAR100_3.00E-03_12_6_1_1000f-03-29-14-00-43.pth'

    SCAFFOLD_name2 = 'results/setting.0401/setting1-3_SCAFFOLD_TOMCNN_CIFAR100_3.00E-03_18_6_1_1000f-03-31-11-36-57.pth'
    FedAvg_name2 = 'results/setting.0401/setting1-3_FedAvg_TOMCNN_CIFAR100_3.00E-03_18_6_1_1000f-03-31-11-36-57.pth'
    SDGT_name2 = 'results/setting.0401/setting1-3_SDGT_TOMCNN_CIFAR100_3.00E-03_18_6_1_1000f-03-31-11-36-57.pth'
    SCAF = torch.load(SCAFFOLD_name)
    FedA = torch.load(FedAvg_name)
    SDGT = torch.load(SDGT_name)

    SCAF2 = torch.load(SCAFFOLD_name2)
    FedA2 = torch.load(FedAvg_name2)
    SDGT2 = torch.load(SDGT_name2)

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(25.6,4.8))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

    # pdb.set_trace()
    axes[0].plot(FedA2[1][0][-1][0][::skip], FedA2[1][0][-1][1][::skip], '--', color = 'blue')
    axes[0].plot(SCAF2[1][0][-1][0][::skip], SCAF2[1][0][-1][1][::skip], ':',  color = 'orange')
    axes[0].plot(SDGT2[1][0][-1][0][::skip], SDGT2[1][0][-1][1][::skip], '-',  color = 'green')
    axes[0].set_xlabel('18 agents, 6 subnets')
    axes[0].grid()

    axes[1].plot(FedA[1][0][2][0][::skip], FedA[1][0][2][1][::skip], '--', color = 'blue')
    axes[1].plot(SCAF[1][0][2][0][::skip], SCAF[1][0][2][1][::skip], ':',  color = 'orange')
    axes[1].plot(SDGT[1][0][2][0][::skip], SDGT[1][0][2][1][::skip], '-',  color = 'green')
    axes[1].set_xlabel('24 agents, 6 subnets')

    axes[1].grid()

    axes[2].plot(FedA[1][0][1][0][::skip], FedA[1][0][1][1][::skip], '--', color = 'blue')
    axes[2].plot(SCAF[1][0][1][0][::skip], SCAF[1][0][1][1][::skip], ':',  color = 'orange')
    axes[2].plot(SDGT[1][0][1][0][::skip], SDGT[1][0][1][1][::skip], '-',  color = 'green')
    axes[2].set_xlabel('36 agents, 6 subnets')
    axes[2].grid()

    axes[3].plot(FedA[1][0][0][0][::skip], FedA[1][0][0][1][::skip], '--', color = 'blue')
    axes[3].plot(SCAF[1][0][0][0][::skip], SCAF[1][0][0][1][::skip], ':',  color = 'orange')
    axes[3].plot(SDGT[1][0][0][0][::skip], SDGT[1][0][0][1][::skip], '-',  color = 'green')
    axes[3].set_xlabel('48 agents, 6 subnets')
    axes[3].grid()

    # pdb.set_trace()

    axes[0].set_title(dataset, loc='right')
    axes[1].set_title(dataset, loc='right')
    axes[2].set_title(dataset, loc='right')
    axes[3].set_title(dataset, loc='right')

    # plt.title(dataset)
    # plt.legend()
    plt.plot(0, 0, '--', color = 'blue', label = 'FedAvg')
    plt.plot(0, 0, ':',  color = 'orange', label = 'SCAFFOLD')
    plt.plot(0, 0, '-',  color = 'green', label = 'SD-GT')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
            ncol=3, fancybox=True)
    plt.ylabel("Test Accuracy", labelpad=10)
    plt.xlabel("Global Aggregation Rounds", labelpad=30)
    save_img(plt, "%s_%s" % (config,dataset), opt)

