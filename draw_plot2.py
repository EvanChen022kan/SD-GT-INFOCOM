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
# SCAFFOLD_name = 'results/07-24/SCAFFOLD_FNN_CIFAR10_3.00E-03_compRAND_acc_2_1f-07-24-11-51-45.pth'
# FedAvg_name = 'results/07-24/FedAvg_FNN_CIFAR10_3.00E-03_compRAND_acc_2_1f-07-24-11-51-45.pth'
# SDGT_name = 'results/07-24/SDGT_FNN_CIFAR10_3.00E-03_compRAND_acc_2_1f-07-24-11-51-45.pth'

'''CIFAR1-60'''
# SCAFFOLD_name = 'results/07-24/SCAFFOLD_FNN_CIFAR10_3.00E-03_compRAND_acc_2_1f-07-24-12-32-48.pth'
# FedAvg_name = 'results/07-24/FedAvg_FNN_CIFAR10_3.00E-03_compRAND_acc_2_1f-07-24-12-32-48.pth'
# SDGT_name = 'results/07-24/SDGT_FNN_CIFAR10_3.00E-03_compRAND_acc_2_1f-07-24-12-32-48.pth'

'''CIFAR10-90'''
# SCAFFOLD_name = 'results/07-26/SCAFFOLD_FNN_CIFAR10_1.00E-03_compRAND_acc_2_1f-07-26-23-49-17.pth'
# FedAvg_name = 'results/07-26/FedAvg_FNN_CIFAR10_1.00E-03_compRAND_acc_2_1f-07-26-23-49-17.pth'
# SDGT_name = 'results/07-26/SDGT_FNN_CIFAR10_1.00E-03_compRAND_acc_2_1f-07-26-23-49-17.pth'


'''CIFAR100'''
# SCAFFOLD_name = 'results/07-25/SCAFFOLD_FNN_CIFAR100_3.00E-03_compRAND_acc_2_1f-07-25-05-31-27.pth'
# FedAvg_name = 'results/07-25/FedAvg_FNN_CIFAR100_3.00E-03_compRAND_acc_2_1f-07-25-05-31-27.pth'
# SDGT_name = 'results/07-25/SDGT_FNN_CIFAR100_3.00E-03_compRAND_acc_2_1f-07-25-05-31-27.pth'

# '''SYN-80'''
# SCAFFOLD_name = 'results/07-27 copy/syn_SCAFFOLD_FNN_MNIST_1.00E-04_acc_5_40f-07-27-11-31-56.pth'
# FedAvg_name = 'results/07-27 copy/syn_FedAvg_FNN_MNIST_1.00E-04_acc_5_40f-07-27-11-31-56.pth'
# SDGT_name = 'results/07-27 copy/syn_SDGT_FNN_MNIST_1.00E-04_acc_5_40f-07-27-11-31-56.pth'


'''After 2024.02'''

dataset = 'CIFAR10'
SCAFFOLD_name1 = 'results/02-11/SCAFFOLD_FNN_CIFAR10_1.00E-03_compRAND_acc_1_1000f-02-11-06-19-00.pth'
FedAvg_name1 = 'results/02-11/FedAvg_FNN_CIFAR10_1.00E-03_compRAND_acc_1_1000f-02-11-06-19-00.pth'
SDGT_name1 = 'results/02-11/SDGT_FNN_CIFAR10_1.00E-03_compRAND_acc_1_1000f-02-11-06-19-00.pth'

SCAFFOLD_name2 = 'results/02-12/SCAFFOLD_FNN_CIFAR10_1.00E-03_compRAND_acc_1_1000f-02-12-05-25-18.pth'
FedAvg_name2 = 'results/02-12/FedAvg_FNN_CIFAR10_1.00E-03_compRAND_acc_1_1000f-02-12-05-25-18.pth'
SDGT_name2 = 'results/02-12/SDGT_FNN_CIFAR10_1.00E-03_compRAND_acc_1_1000f-02-12-05-25-18.pth'


legend_list = []

SCAF1 = torch.load(SCAFFOLD_name1)
FedA1 = torch.load(FedAvg_name1)
SDGT1 = torch.load(SDGT_name1)
SCAF2 = torch.load(SCAFFOLD_name2)
FedA2 = torch.load(FedAvg_name2)
SDGT2 = torch.load(SDGT_name2)


skip = 16
skip = 4
skip = 1

K = 10
plt.plot(FedA1[0][0][::skip], FedA1[0][1][::skip], '--', color = 'blue')
plt.plot(SCAF1[0][0][::skip], SCAF1[0][1][::skip], ':', color = 'orange')
plt.plot(SDGT1[0][0][::skip], SDGT1[0][1][::skip], '-', color = 'green')
# plt.plot(FedA[i][0][::skip], FedA[i][1][::skip], '--')
# plt.plot(SCAF[i][0][::skip], SCAF[i][1][::skip], ':')
# plt.plot(SDGT[i][0][::skip], SDGT[i][1][::skip], '-')
legend_list.append('FedAvg')
legend_list.append('SCAFFOLD')
legend_list.append('SD-GT')
# plt.legend(legend_list)
save_img(plt, "FINAL_noleg_K-%d_%s" % (K, dataset), opt)
plt.clf()
legend_list = []

K = 3
plt.plot(FedA1[1][0][::skip], FedA1[1][1][::skip], '--', color = 'blue')
plt.plot(SCAF1[1][0][::skip], SCAF1[1][1][::skip], ':', color = 'orange')
plt.plot(SDGT1[1][0][::skip], SDGT1[1][1][::skip], '-', color = 'green')
# plt.plot(FedA[i][0][::skip], FedA[i][1][::skip], '--')
# plt.plot(SCAF[i][0][::skip], SCAF[i][1][::skip], ':')
# plt.plot(SDGT[i][0][::skip], SDGT[i][1][::skip], '-')
legend_list.append('FedAvg')
legend_list.append('SCAFFOLD')
legend_list.append('SD-GT')
# plt.legend(legend_list)
save_img(plt, "FINAL_noleg_K-%d_%s" % (K, dataset), opt)
plt.clf()
legend_list = []

K = 1
plt.plot(FedA1[2][0][::skip], FedA1[2][1][::skip], '--', color = 'blue')
plt.plot(SCAF1[2][0][::skip], SCAF1[2][1][::skip], ':', color = 'orange')
plt.plot(SDGT1[2][0][::skip], SDGT1[2][1][::skip], '-', color = 'green')
# plt.plot(FedA[i][0][::skip], FedA[i][1][::skip], '--')
# plt.plot(SCAF[i][0][::skip], SCAF[i][1][::skip], ':')
# plt.plot(SDGT[i][0][::skip], SDGT[i][1][::skip], '-')
legend_list.append('FedAvg')
legend_list.append('SCAFFOLD')
legend_list.append('SD-GT')
# plt.legend(legend_list)
save_img(plt, "FINAL_noleg_K-%d_%s" % (K, dataset), opt)
plt.clf()
legend_list = []

K = 40
plt.plot(FedA2[0][0][::skip], FedA2[0][1][::skip], '--', color = 'blue')
plt.plot(SCAF2[0][0][::skip], SCAF2[0][1][::skip], ':', color = 'orange')
plt.plot(SDGT2[0][0][::skip], SDGT2[0][1][::skip], '-', color = 'green')
# plt.plot(FedA[i][0][::skip], FedA[i][1][::skip], '--')
# plt.plot(SCAF[i][0][::skip], SCAF[i][1][::skip], ':')
# plt.plot(SDGT[i][0][::skip], SDGT[i][1][::skip], '-')
legend_list.append('FedAvg')
legend_list.append('SCAFFOLD')
legend_list.append('SD-GT')
# plt.legend(legend_list)
save_img(plt, "FINAL_noleg_K-%d_%s" % (K, dataset), opt)
plt.clf()
legend_list = []

K = 20
plt.plot(FedA2[1][0][::skip], FedA2[1][1][::skip], '--', color = 'blue')
plt.plot(SCAF2[1][0][::skip], SCAF2[1][1][::skip], ':', color = 'orange')
plt.plot(SDGT2[1][0][::skip], SDGT2[1][1][::skip], '-', color = 'green')
# plt.plot(FedA[i][0][::skip], FedA[i][1][::skip], '--')
# plt.plot(SCAF[i][0][::skip], SCAF[i][1][::skip], ':')
# plt.plot(SDGT[i][0][::skip], SDGT[i][1][::skip], '-')
legend_list.append('FedAvg')
legend_list.append('SCAFFOLD')
legend_list.append('SD-GT')
# plt.legend(legend_list)
save_img(plt, "FINAL_noleg_K-%d_%s" % (K, dataset), opt)
plt.clf()
pdb.set_trace()

K = [1, 3, 10]


fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)

# pdb.set_trace()
axes[0].plot(FedA[-2][0][::skip], FedA[-2][1][::skip], '--', color = 'blue')
axes[0].plot(SCAF[-2][0][::skip], SCAF[-2][1][::skip], ':',  color = 'orange')
axes[0].plot(SDGT[-2][0][::skip], SDGT[-2][1][::skip], '-',  color = 'green')
axes[0].grid()

axes[1].plot(FedA[-3][0][::skip], FedA[-3][1][::skip], '--', color='red')
axes[1].plot(SCAF[-3][0][::skip], SCAF[-3][1][::skip], ':', color='purple')
axes[1].plot(SDGT[-3][0][::skip], SDGT[-3][1][::skip], '-', color='brown')
# axes[1].plot(FedA[-1][0][::skip], FedA[-1][1][::skip], '--', color='red')
# axes[1].plot(SCAF[-1][0][::skip], SCAF[-1][1][::skip], ':', color='purple')
# axes[1].plot(SDGT[-1][0][::skip], SDGT[-1][1][::skip], '-', color='brown')
axes[1].grid()

# axes[0].set_title(dataset + ' (K = %d)' % K[-2], loc='right')
# axes[1].set_title(dataset + ' (K = %d)' % K[-1], loc='right')
plt.title(dataset)
plt.ylabel("Test Accuracy")
plt.xlabel("Global Aggregation Rounds")
save_img(plt, "FINAL_noleg_%s" % (dataset), opt)

# axes[0].set_title(dataset + r", $\frac{h_{c_i}}{m_{c_i}} = 20 /%$")
# axes[1].set_title(dataset + r", $\frac{h_{c_i}}{m_{c_i}} = 100 /%$")
# plt.ylabel(r"$||x_{g}^t - x^{*}||_2$")
# plt.xlabel("Global Aggregation Rounds")
# save_img(plt, "SYN_FINAL_noleg_%s" % (dataset), opt)

# for i in range(1, len(K)):
#     # plt.subplot(1, 2, i)
#     plt.plot(FedA[-(i+1)][0][::skip], FedA[-(i+1)][1][::skip], '--')
#     plt.plot(SCAF[-(i+1)][0][::skip], SCAF[-(i+1)][1][::skip], ':')
#     plt.plot(SDGT[-(i+1)][0][::skip], SDGT[-(i+1)][1][::skip], '-')
#     # plt.plot(FedA[i][0][::skip], FedA[i][1][::skip], '--')
#     # plt.plot(SCAF[i][0][::skip], SCAF[i][1][::skip], ':')
#     # plt.plot(SDGT[i][0][::skip], SDGT[i][1][::skip], '-')
#     legend_list.append('FedAvg, K = %d' % K[i])
#     legend_list.append('SCAFFOLD, K = %d' % K[i])
#     legend_list.append('SD-GT, K = %d' % K[i])





# plt.title('%s' % dataset, loc='right')
# plt.ylabel("Test Accuracy")
# plt.xlabel("Global Aggregation Rounds")
# plt.grid()
# save_img(plt, "FINAL_noleg_%s" % (dataset), opt)
# # plt.legend(legend_list)
# # save_img(plt, "FINAL_%s" % (dataset), opt)

