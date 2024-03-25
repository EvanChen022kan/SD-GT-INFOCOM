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
SCAFFOLD_name = 'results/07-14/SCAFFOLD_FNN_MNIST_1.00E-02_compRAND_acc_2_1f-07-14-18-51-17.pth'
# SCAFFOLD_name = 'results/07-15/SCAFFOLD_FNN_CIFAR10_3.00E-03_compRAND_acc_2_1f-07-15-05-01-42.pth'
# SCAFFOLD_name = 'results/07-16/SCAFFOLD_FNN_CIFAR100_3.00E-03_compRAND_acc_2_1f-07-16-03-19-05.pth'



FedAvg_name = 'results/07-14/FedAvg_FNN_MNIST_1.00E-02_compRAND_acc_2_1f-07-14-18-51-17.pth'
# FedAvg_name = 'results/07-15/FedAvg_FNN_CIFAR10_3.00E-03_compRAND_acc_2_1f-07-15-05-01-42.pth'
# FedAvg_name = 'results/07-16/FedAvg_FNN_CIFAR100_3.00E-03_compRAND_acc_2_1f-07-16-03-19-05.pth'


SDGT_name = 'results/07-14/SDGT_FNN_MNIST_1.00E-02_compRAND_acc_2_1f-07-14-18-51-17.pth'
# SDGT_name = 'results/07-15/SDGT_FNN_CIFAR10_3.00E-03_compRAND_acc_2_1f-07-15-05-01-42.pth'
# SDGT_name = 'results/07-16/SDGT_FNN_CIFAR100_3.00E-03_compRAND_acc_2_1f-07-16-03-19-05.pth'

SCAFFOLD_name = 'results/07-23/SCAFFOLD_FNN_MNIST_1.00E-02_compRAND_acc_2_1f-07-23-23-09-35.pth'
FedAvg_name = 'results/07-23/FedAvg_FNN_MNIST_1.00E-02_compRAND_acc_2_1f-07-23-23-09-35.pth'
SDGT_name = 'results/07-23/SDGT_FNN_MNIST_1.00E-02_compRAND_acc_2_1f-07-23-23-09-35.pth'

'''CIFAR1-30'''


'''CIFAR1-60'''


'''CIFAR10-90'''



'''CIFAR100'''



'''After 2024.02'''

dataset = 'CIFAR10'
SCAFFOLD_name90_3 = 'results/02-18/SCAFFOLD_FNN_CIFAR10_1.00E-03_compRAND_acc_90_3_1_1000f-02-18-23-45-24.pth'
FedAvg_name90_3 = 'results/02-18/FedAvg_FNN_CIFAR10_1.00E-03_compRAND_acc_90_3_1_1000f-02-18-23-45-24.pth'
SDGT_name90_3 = 'results/02-18/SDGT_FNN_CIFAR10_1.00E-03_compRAND_acc_90_3_1_1000f-02-18-23-45-24.pth'

SCAFFOLD_name90_6 = 'results/02-11/SCAFFOLD_FNN_CIFAR10_1.00E-03_compRAND_acc_1_1000f-02-11-19-24-20.pth'
FedAvg_name90_6 = 'results/02-11/FedAvg_FNN_CIFAR10_1.00E-03_compRAND_acc_1_1000f-02-11-19-24-20.pth'
SDGT_name90_6 = 'results/02-11/SDGT_FNN_CIFAR10_1.00E-03_compRAND_acc_1_1000f-02-11-19-24-20.pth'

SCAFFOLD_name90_18 = 'results/02-12/SCAFFOLD_FNN_CIFAR10_1.00E-03_compRAND_acc_90_18_1_1000f-02-12-10-21-44.pth'
FedAvg_name90_18 = 'results/02-12/FedAvg_FNN_CIFAR10_1.00E-03_compRAND_acc_90_18_1_1000f-02-12-10-21-44.pth'
SDGT_name90_18 = 'results/02-12/SDGT_FNN_CIFAR10_1.00E-03_compRAND_acc_90_18_1_1000f-02-12-10-21-44.pth'

SCAFFOLD_name90_30 = 'results/02-13/SCAFFOLD_FNN_CIFAR10_1.00E-03_compRAND_acc_90_30_1_1000f-02-13-00-53-47.pth'
FedAvg_name90_30 = 'results/02-13/FedAvg_FNN_CIFAR10_1.00E-03_compRAND_acc_90_30_1_1000f-02-13-00-53-47.pth'
SDGT_name90_30 = 'results/02-13/SDGT_FNN_CIFAR10_1.00E-03_compRAND_acc_90_30_1_1000f-02-13-00-53-47.pth'



legend_list = []






skip = 16
skip = 4
skip = 1

K = 10
SCAF1 = torch.load(SCAFFOLD_name90_3)
FedA1 = torch.load(FedAvg_name90_3)
SDGT1 = torch.load(SDGT_name90_3)
plt.plot(FedA1[0][0][::skip], FedA1[0][1][::skip], '--', color = 'blue')
plt.plot(SCAF1[0][0][::skip], SCAF1[0][1][::skip], ':', color = 'orange')
plt.plot(SDGT1[0][0][::skip], SDGT1[0][1][::skip], '-', color = 'green')
legend_list.append('FedAvg')
legend_list.append('SCAFFOLD')
legend_list.append('SD-GT')
plt.grid()
plt.gca().set_ylim(top=0.6)

# plt.legend(legend_list)
save_img(plt, "comp_n_90_3_%s" % (dataset), opt)


plt.clf()
legend_list = []


SCAF1 = torch.load(SCAFFOLD_name90_6)
FedA1 = torch.load(FedAvg_name90_6)
SDGT1 = torch.load(SDGT_name90_6)
plt.plot(FedA1[0][0][::skip], FedA1[0][1][::skip], '--', color = 'blue')
plt.plot(SCAF1[0][0][::skip], SCAF1[0][1][::skip], ':', color = 'orange')
plt.plot(SDGT1[0][0][::skip], SDGT1[0][1][::skip], '-', color = 'green')
legend_list.append('FedAvg')
legend_list.append('SCAFFOLD')
legend_list.append('SD-GT')
plt.grid()
plt.gca().set_ylim(top=0.6)

# plt.legend(legend_list)
save_img(plt, "comp_n_90_6_%s" % (dataset), opt)
plt.clf()
legend_list = []

SCAF1 = torch.load(SCAFFOLD_name90_18)
FedA1 = torch.load(FedAvg_name90_18)
SDGT1 = torch.load(SDGT_name90_18)
plt.plot(FedA1[0][0][::skip], FedA1[0][1][::skip], '--', color = 'blue')
plt.plot(SCAF1[0][0][::skip], SCAF1[0][1][::skip], ':', color = 'orange')
plt.plot(SDGT1[0][0][::skip], SDGT1[0][1][::skip], '-', color = 'green')
legend_list.append('FedAvg')
legend_list.append('SCAFFOLD')
legend_list.append('SD-GT')
plt.grid()
plt.gca().set_ylim(top=0.6)

# plt.legend(legend_list)
save_img(plt, "comp_n_90_18_%s" % (dataset), opt)

plt.clf()
legend_list = []

SCAF1 = torch.load(SCAFFOLD_name90_30)
FedA1 = torch.load(FedAvg_name90_30)
SDGT1 = torch.load(SDGT_name90_30)
plt.plot(FedA1[0][0][::skip], FedA1[0][1][::skip], '--', color = 'blue')
plt.plot(SCAF1[0][0][::skip], SCAF1[0][1][::skip], ':', color = 'orange')
plt.plot(SDGT1[0][0][::skip], SDGT1[0][1][::skip], '-', color = 'green')
legend_list.append('FedAvg')
legend_list.append('SCAFFOLD')
legend_list.append('SD-GT')
plt.grid()
plt.gca().set_ylim(top=0.6)

# plt.legend(legend_list)
save_img(plt, "comp_n_90_30_%s" % (dataset), opt)
plt.clf()
legend_list = []
