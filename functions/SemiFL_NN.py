import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .gradfn import func
from .utils import get_mse_fstar, build_asy_list
from .NN import twoLAYER_NN, Res_NN, ModelCNN, TOMCNN
from torchvision import transforms
import random
import scipy.io as sio
import copy
from collections import OrderedDict
from .GT_func import *
import pdb


def SemiFL_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, step_c, T=1, n_agents=10, opt=None):
    N = input.shape[0]
    dim = input.shape[-1]
    device = torch.device('cuda:{}'.format(opt.gpu_id))
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    label_list = list(torch.split(labels.squeeze(), n, dim=0))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    # transform_train = nn.Identity()
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)
    # c = list(zip(input_list, label_list))
    # random.shuffle(c)
    # input_list, label_list = zip(*c)

    if opt.model == 'FNN':
        f = [twoLAYER_NN(dim, opt) for i in range(n_agents)]
        global_f = twoLAYER_NN(dim, opt)
    elif opt.model == 'CNN2':
        f = [ModelCNN(opt) for i in range(n_agents)]
        global_f = ModelCNN(opt)
    elif opt.model == 'TOMCNN':
        f = [TOMCNN(opt) for i in range(n_agents)]
        global_f = TOMCNN(opt)
    else:
        f = [Res_NN(dim, opt) for i in range(n_agents)]
        global_f = Res_NN(dim, opt)


    x1_list = [Variable(torch.rand(100, dim, device=device), requires_grad=True) for i in range(n_agents)]
    x2_list = [Variable(torch.rand(10, 100, device=device), requires_grad=True) for i in range(n_agents)]



    '''initialization for all f'''
    central_f_dict = global_f.state_dict()
    for i in range(n_agents):
        f[i].load_state_dict(central_f_dict)
    z_i = OrderedDict()
    # y_i = {}
    for name in f[0].state_dict():
        z_i[name] = torch.zeros(f[0].state_dict()[name].shape).to(device)
        # y_i[name] = torch.zeros(param.shape).to(device)

    # y = [copy.deepcopy(z_i) for i in range(n_agents)]
    # z = [copy.deepcopy(z_i) for i in range(n_agents)]
    x_new = [copy.deepcopy(z_i) for i in range(n_agents)]
    # x1_new = [0 for i in range(n_agents)]
    # x2_new = [0 for i in range(n_agents)]


    com_count = 0
    gamma = step_c

    error_list = []
    acc_list = []

    iter_list = []

    lossfn = nn.CrossEntropyLoss()

    com_iter = 0
    iters = 0

    '''evaluate'''
    result = []
    test_counter = 0
    with torch.no_grad():
        while (test_counter + opt.batch <= len(t_input)):
            test_data = t_input[test_counter: test_counter+opt.batch]
            if opt.model == 'FNN':
                test_data = test_data.flatten(start_dim=1)
            output = global_f(test_data)
            result.append(output)
            test_counter += opt.batch
    result = torch.cat(result, dim=0)
    # pdb.set_trace()
    loss = lossfn(result, t_labels[:len(result)])
    t1 = torch.argmax(F.softmax(result, dim=1), dim=1)
    acc = 0
    for i in range(len(t1)):
        if t1[i] == t_labels[i]:
            acc += 1
    acc_list.append(acc/len(t1))
    # pdb.set_trace()
    iter_list.append(iters)
    error_list.append(loss.item())
    print("[SemiFL][%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r', end = '')

    # while (com_count < opt.c_rounds):
    nc_agents = n_agents//opt.n_cluster
    # T = int(opt.p_inv)
    data_counter = [0 for i in range(n_agents)]
    for r in range(int(opt.c_rounds)):

        for t in range(T):
            for i in range(n_agents):

                if data_counter[i] + opt.batch <= len(input_list[i]):
                    i_input = input_list[i][data_counter[i]:data_counter[i] + opt.batch]
                    i_label = label_list[i][data_counter[i]:data_counter[i] + opt.batch]
                    data_counter[i] = (data_counter[i] + opt.batch) % len(input_list[i])
                else:
                    i_input1 = input_list[i][data_counter[i]:-1]
                    i_label1 = label_list[i][data_counter[i]:-1]
                    length = opt.batch - i_input1.shape[0]
                    i_input = torch.cat([input_list[i][0: length], i_input1], dim=0)
                    i_label = torch.cat([label_list[i][0: length], i_label1], dim=0)
                    data_counter[i] = (data_counter[i] + opt.batch + 1) % len(input_list[i])
                # data_counter[i] = (data_counter[i] + opt.batch) % len(input_list[i] - opt.batch)
                if opt.dataset != 'MNIST':
                    i_input = transform_train(i_input)
                if opt.model == 'FNN':
                    i_input = i_input.flatten(start_dim=1)
                f[i].zero_grad()
                output = f[i](i_input)
                loss = lossfn(output, i_label)
                loss.backward()


                # pdb.set_trace()
                # print(data_counter[i])
                with torch.no_grad():
                    new_param = OrderedDict()
                    t1 = f[i].state_dict()
                    for name in f[i].state_dict():
                        if ('num_batches_tracked' in name or 'running_var' in name or 'running_mean' in name):
                            del t1[name]
                    grad_dict = {k: v.grad for k, v in zip(t1, f[i].parameters())}
                    for name in f[i].state_dict():
                        if ('num_batches_tracked' in name or 'running_var' in name or 'running_mean' in name):
                            new_param[name] = f[i].state_dict()[name]
                        else:
                            new_param[name] = f[i].state_dict()[name] - gamma*grad_dict[name]
                    # f[i].linear1.weight = torch.nn.Parameter(f[i].linear1.weight - gamma*f[i].linear1.weight.grad)
                    # f[i].linear2.weight = torch.nn.Parameter(f[i].linear2.weight - gamma*f[i].linear2.weight.grad)
                    f[i].load_state_dict(new_param)

                # pdb.set_trace()

                # pdb.set_trace()


            with torch.no_grad():
                for i in range(n_agents):
                    cluster = i//nc_agents
                    ic = i % nc_agents
                    # x1_new[i] = R[cluster][ic, ic]*f[i].linear1.weight
                    # x2_new[i] = R[cluster][ic, ic]*f[i].linear2.weight
                    # for j in N_in[cluster][ic]:
                    #     x1_new[i] += R[cluster][ic, j]*f[cluster*nc_agents + j].linear1.weight
                    #     x2_new[i] += R[cluster][ic, j]*f[cluster*nc_agents + j].linear2.weight
                    x_new[i] = update_xnew(f, R, N_in, i, nc_agents, x_new[i])

                for i in range(n_agents):
                    # f[i].linear1.weight = torch.nn.Parameter(x1_new[i])
                    # f[i].linear2.weight = torch.nn.Parameter(x2_new[i])
                    f[i].load_state_dict(x_new[i])

            # pdb.set_trace()
            
            # print(data_counter[i])

        # com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        com_list = [item for sublist in com_list for item in sublist]
        '''aggregate'''
        x_cen = copy.deepcopy(z_i)
        # x_cen2 = 0
        with torch.no_grad():
            for n_c in com_list:
                for name in global_f.state_dict():
                    x_cen[name] += 1/len(com_list)*f[n_c].state_dict()[name]
                # x_cen1 += 1/len(com_list)*f[n_c].linear1.weight
                # x_cen2 += 1/len(com_list)*f[n_c].linear2.weight

            '''broadcast'''
            for n_c in com_list:
                '''correction term update'''
                f[n_c].load_state_dict(x_cen)
                # f[n_c].linear1.weight = torch.nn.Parameter(x_cen1)
                # f[n_c].linear2.weight = torch.nn.Parameter(x_cen2)
                # pdb.set_trace()

        iters += 1
        com_iter += 1
        com_count += 1

        # print("[%d agents]: " % n_agents, '\tIters:', iters, '\t\t\t\r')
        if iters % 10 == 0:

            # pdb.set_trace()
            '''compute loss'''
            # x_cen = copy.deepcopy(z_i)
            # for i in range(n_agents):
            #     for name in global_f.state_dict():
            #         x_cen[name] += 1/n_agents*f[i].state_dict()[name]


            # global_f.linear1.weight = torch.nn.Parameter(x_cen1)
            # global_f.linear2.weight = torch.nn.Parameter(x_cen2)
            global_f.load_state_dict(x_cen)
            result = []
            test_counter = 0
            with torch.no_grad():
                while(test_counter + opt.batch <= len(t_input)):
                    test_data = t_input[test_counter: test_counter+opt.batch]
                    if opt.model == 'FNN':
                        test_data = test_data.flatten(start_dim=1)
                    output = global_f(test_data)
                    result.append(output)
                    test_counter += opt.batch
            result = torch.cat(result, dim= 0)
            # pdb.set_trace()
            loss = lossfn(result, t_labels[:len(result)])
            t1 = torch.argmax(F.softmax(result, dim=1), dim=1)
            acc = 0
            for i in range(len(t1)):
                if t1[i] == t_labels[i]:
                    acc += 1
            acc_list.append(acc/len(t1))
            # pdb.set_trace()
            iter_list.append(iters)
            error_list.append(loss.item())
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % 100 == 0:
            # print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            print("[SemiFL][%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], "\tAcc: %.2f" % (100*acc_list[-1]), '\t\t\t\t\r', end = '')

            # pdb.set_trace()
    print('\n', end = '')
    return iter_list, error_list, acc_list


def SemiFLGT_NN(input, labels, R, C, N_in, N_out, step_c, T=1, n_agents=10, opt=None):
    N = input.shape[0]
    dim = input.shape[1]
    device = torch.device('cuda:{}'.format(opt.gpu_id))
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    label_list = list(torch.split(labels.squeeze(), n, dim=0))
    # pdb.set_trace()
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)

    f = [twoLAYER_NN(opt) for i in range(n_agents)]
    global_f = twoLAYER_NN(opt)

    # x1_list = [Variable(torch.rand(100, 28*28, device=device), requires_grad=True) for i in range(n_agents)]
    # x2_list = [Variable(torch.rand(10, 100, device=device), requires_grad=True) for i in range(n_agents)]

    x1_new = [0 for i in range(n_agents)]
    x2_new = [0 for i in range(n_agents)]

    y1 = [0 for i in range(n_agents)]
    y2 = [0 for i in range(n_agents)]



    com_count = 0
    gamma = step_c

    error_list = []
    iter_list = []

    lossfn = nn.CrossEntropyLoss()

    com_iter = 0
    iters = 0

    '''evaluate'''
    result = []
    test_counter = 0
    with torch.no_grad():
        while (test_counter + opt.batch <= len(input)):
            test_data = input[test_counter: test_counter+opt.batch]
            output = global_f(test_data)
            result.append(output)
            test_counter += opt.batch
    result = torch.cat(result, dim=0)
    # pdb.set_trace()
    loss = lossfn(result, labels[:len(result)])
    # pdb.set_trace()
    iter_list.append(iters)
    error_list.append(loss.item())
    print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')

    # while (com_count < opt.c_rounds):
    nc_agents = n_agents//opt.n_cluster
    T = int(opt.p_inv)
    data_counter = [0 for i in range(n_agents)]
    for r in range(int(opt.c_rounds)):

        for t in range(T):
            for i in range(n_agents):

                i_input = input_list[i][data_counter[i]:data_counter[i] + opt.batch]
                i_label = label_list[i][data_counter[i]:data_counter[i] + opt.batch]
                # data_counter[i] = (data_counter[i] + opt.batch) % len(input_list[i] - opt.batch)
                output = f[i](i_input)
                loss = lossfn(output, i_label)
                loss.backward()

                # pdb.set_trace()
                # print(data_counter[i])
                with torch.no_grad():
                    f[i].linear1.weight = torch.nn.Parameter(f[i].linear1.weight - gamma*(f[i].linear1.weight.grad + y1[i]))
                    f[i].linear2.weight = torch.nn.Parameter(f[i].linear2.weight - gamma*(f[i].linear2.weight.grad + y2[i]))

                # pdb.set_trace()

                # pdb.set_trace()

            with torch.no_grad():
                for i in range(n_agents):
                    cluster = i//nc_agents
                    ic = i % nc_agents
                    x1_new[i] = R[cluster][ic, ic]*f[i].linear1.weight
                    x2_new[i] = R[cluster][ic, ic]*f[i].linear2.weight
                    for j in N_in[cluster][ic]:
                        x1_new[i] += R[cluster][ic, j]*f[cluster*nc_agents + j].linear1.weight
                        x2_new[i] += R[cluster][ic, j]*f[cluster*nc_agents + j].linear2.weight

                for i in range(n_agents):
                    f[i].linear1.weight = torch.nn.Parameter(x1_new[i])
                    f[i].linear2.weight = torch.nn.Parameter(x2_new[i])

            # pdb.set_trace()

            # print(data_counter[i])

        com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        '''aggregate'''
        x_cen1 = 0
        x_cen2 = 0
        with torch.no_grad():
            for n_c in com_list:
                x_cen1 += 1/opt.n_cluster*(f[n_c].linear1.weight + T*gamma*y1[n_c])
                x_cen2 += 1/opt.n_cluster*(f[n_c].linear2.weight + T*gamma*y2[n_c])

            '''broadcast'''
            for n_c in com_list:
                y1[n_c] = y1[n_c] + 1/(T*gamma)*(f[n_c].linear1.weight - x_cen1)
                y2[n_c] = y2[n_c] + 1/(T*gamma)*(f[n_c].linear2.weight - x_cen2)
                '''correction term update'''
                f[n_c].linear1.weight = torch.nn.Parameter(x_cen1)
                f[n_c].linear2.weight = torch.nn.Parameter(x_cen2)
                # pdb.set_trace()

        iters += 1
        com_iter += 1
        com_count += 1

        # print("[%d agents]: " % n_agents, '\tIters:', iters, '\t\t\t\r')
        if iters % 50 == 0:

            # pdb.set_trace()
            '''compute loss'''
            x_cen1 = 0
            x_cen2 = 0
            for i in range(n_agents):
                x_cen1 += 1/n_agents*f[i].linear1.weight
                x_cen2 += 1/n_agents*f[i].linear2.weight

            global_f.linear1.weight = torch.nn.Parameter(x_cen1)
            global_f.linear2.weight = torch.nn.Parameter(x_cen2)
            result = []
            test_counter = 0
            with torch.no_grad():
                while (test_counter + opt.batch <= len(input)):
                    test_data = input[test_counter: test_counter+opt.batch]
                    output = global_f(test_data)
                    result.append(output)
                    test_counter += opt.batch
            result = torch.cat(result, dim=0)
            # pdb.set_trace()
            loss = lossfn(result, labels[:len(result)])
            # pdb.set_trace()
            iter_list.append(iters)
            error_list.append(loss.item())
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % 100 == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            # pdb.set_trace()

    return iter_list, error_list


def SemiFLGT2_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, step_c, T=1, sample_num_opt=1, n_agents=10, control=0, lam = [], energy_list = [], opt=None):
    N = input.shape[0]
    dim = input.shape[-1]
    device = torch.device('cuda:{}'.format(opt.gpu_id))
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    label_list = list(torch.split(labels.squeeze(), n, dim=0))
    nc_agents = n_agents//opt.n_cluster

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    # transform_train = nn.Identity()


    # pdb.set_trace()
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)
    # c = list(zip(input_list, label_list))
    # random.shuffle(c)
    # input_list, label_list = zip(*c)


    if opt.model == 'FNN':
        f = [twoLAYER_NN(dim, opt) for i in range(n_agents)]
        global_f = twoLAYER_NN(dim, opt)
    elif opt.model == 'CNN2':
        f = [ModelCNN(opt) for i in range(n_agents)]
        global_f = ModelCNN(opt)
    elif opt.model == 'TOMCNN':
        f = [TOMCNN(opt) for i in range(n_agents)]
        global_f = TOMCNN(opt)
    else:
        f = [Res_NN(dim, opt) for i in range(n_agents)]
        global_f = Res_NN(dim, opt)
    # pdb.set_trace()

    '''initialization of z, y'''
    z_i = OrderedDict()
    # y_i = {}
    for name in f[0].state_dict():
        z_i[name] = torch.zeros(f[0].state_dict()[name].shape).to(device)
        # y_i[name] = torch.zeros(param.shape).to(device)
    

    '''initilization of all agents'''
    center_f_dict = global_f.state_dict()
    x_cen = copy.deepcopy(center_f_dict)
    # x_cen = copy.deepcopy(z_i)
    for i in range(n_agents):
        f[i].load_state_dict(center_f_dict)
    y = [copy.deepcopy(z_i) for i in range(n_agents)]
    z = [copy.deepcopy(z_i) for i in range(n_agents)]
    z_new = [copy.deepcopy(z_i) for i in range(n_agents)]
    x_init = [copy.deepcopy(z_i) for i in range(n_agents)]
    x_old = [copy.deepcopy(z_i) for i in range(n_agents)]
    x_new = [copy.deepcopy(z_i) for i in range(n_agents)]
    cluster_y = [copy.deepcopy(z_i) for i in range(n_agents//nc_agents)]
    old_cluster_y = [copy.deepcopy(z_i) for i in range(n_agents//nc_agents)]

    # pdb.set_trace()
    


    # x1_new = [0 for i in range(n_agents)]
    # x2_new = [0 for i in range(n_agents)]

    # y1 = [0 for i in range(n_agents)]
    # y2 = [0 for i in range(n_agents)]

    com_count = 0
    gamma = step_c

    error_list = []
    acc_list = []
    iter_list = []

    lossfn = nn.CrossEntropyLoss()

    com_iter = 0
    iters = 0

    '''evaluate'''
    result = []
    test_counter = 0
    with torch.no_grad():
        while (test_counter + opt.batch <= len(t_input)):
            test_data = t_input[test_counter: test_counter+opt.batch]
            if opt.model == 'FNN':
                test_data = test_data.flatten(start_dim=1)
            output = global_f(test_data)
            result.append(output)
            test_counter += opt.batch
    result = torch.cat(result, dim=0)
    # pdb.set_trace()
    loss = lossfn(result, t_labels[:len(result)])
    t1 = torch.argmax(F.softmax(result, dim=1), dim=1)
    acc= 0
    for i in range(len(t1)):
        if t1[i] == t_labels[i]:
            acc += 1
    acc_list.append(acc/len(t1))
    # pdb.set_trace()
    iter_list.append(iters)
    error_list.append(loss.item())
    print("[%d,%d][%d agents]: " % (T, opt.sample_num, n_agents), '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r', end = '')


    x_old1 = [0 for i in range(n_agents)]
    # x_old2 = [0 for i in range(n_agents)]
    # z1 = [0 for i in range(n_agents)]
    # z2 = [0 for i in range(n_agents)]
    # z_new1 = [0 for i in range(n_agents)]
    # z_new2 = [0 for i in range(n_agents)]
    # cluster_y1 = [0 for i in range(n_agents)]
    # cluster_y2 = [0 for i in range(n_agents)]


    # x_init1 = [0 for i in range(n_agents)]
    # x_init2 = [0 for i in range(n_agents)]
    # x_cen1 = 0
    # x_cen2 = 0


    # while (com_count < opt.c_rounds):
    # T = int(opt.p_inv)
    data_counter = [0 for i in range(n_agents)]
    # energy_list = [random.randint(1, 100) for i in range(opt.n_cluster)]
    energy_count = 0
    energy_cost_list = []

    sample_num = np.zeros(opt.c_rounds+1)
    sample_num[0] = opt.sample_num
    K_list = np.zeros(opt.c_rounds+1)
    K_list[0] = T
    p = np.zeros(opt.c_rounds+1)
    p[0] = 1 - ((nc_agents - opt.sample_num)/nc_agents)**2

    for r in range(int(opt.c_rounds)):
        for i in range(n_agents):
            # z_new1[i] = z1[i]
            # z_new2[i] = z2[i]
            z_new[i] = copy.deepcopy(z[i])
            # x_init1[i] = f[i].linear1.weight
            # x_init2[i] = f[i].linear2.weight
            x_init[i] = copy.deepcopy(f[i].state_dict())


        for t in range(T):
            # print("before")
            # print(y[0]['linear1.weight'])
            for i in range(n_agents):

                if data_counter[i] + opt.batch <= len(input_list[i]):
                    i_input = input_list[i][data_counter[i]:data_counter[i] + opt.batch]
                    i_label = label_list[i][data_counter[i]:data_counter[i] + opt.batch]
                    data_counter[i] = (data_counter[i] + opt.batch) % len(input_list[i])
                else:
                    i_input1 = input_list[i][data_counter[i]:-1]
                    i_label1 = label_list[i][data_counter[i]:-1]
                    length = opt.batch - i_input1.shape[0]
                    i_input = torch.cat([input_list[i][0: length], i_input1], dim=0)
                    i_label = torch.cat([label_list[i][0: length], i_label1], dim=0)
                    data_counter[i] = (data_counter[i] + opt.batch + 1) % len(input_list[i])
                if opt.dataset != 'MNIST':
                    i_input = transform_train(i_input)
                if opt.model == 'FNN':
                    i_input = i_input.flatten(start_dim=1)
                f[i].zero_grad()
                output = f[i](i_input)
                loss = lossfn(output, i_label)
                loss.backward()

                # if i == 0:
                #     print(f[0].linear2.weight)
                with torch.no_grad():
                    # x_old1[i] = f[i].linear1.weight
                    # x_old2[i] = f[i].linear2.weight
                    x_old[i] = copy.deepcopy(f[i].state_dict())
                    # if i == n_agents-1:
                    #     pdb.set_trace()
                    new_param = gradient_update(f[i], y[i], z[i], gamma)
                    # grad_dict = {k: v.grad for k, v in zip(f[i].state_dict(), f[i].parameters())}
                    # if i == 0:
                    #     print(grad_dict['linear2.weight'])
                    # f[i].linear1.weight = torch.nn.Parameter(f[i].linear1.weight - gamma*(f[i].linear1.weight.grad + y[i]['linear1.weight'] + z[i]['linear1.weight']))
                    # f[i].linear2.weight = torch.nn.Parameter(f[i].linear2.weight - gamma*(f[i].linear2.weight.grad + y[i]['linear2.weight'] + z[i]['linear2.weight']))
                    f[i].load_state_dict(new_param)
                    
                # pdb.set_trace()

                # if torch.sum(t1['linear1.weight'] - f[i].linear1.weight) != 0:
            # print(f[0].linear2.weight)

            with torch.no_grad():
                for i in range(n_agents):
                    cluster = i//nc_agents
                    ic = i % nc_agents
                    z_new[i] = update_znew(f, x_old, y, R, N_in, i, nc_agents, T, gamma, z_new[i])


                    # z_r1 = R[cluster][ic, ic]*(f[i].linear1.weight - x_old1[i] + gamma*y1[i])
                    # z_r2 = R[cluster][ic, ic]*(f[i].linear2.weight - x_old2[i] + gamma*y2[i])

                    # for j in N_in[cluster][ic]:
                    #     z_r1 += R[cluster][ic, j]*(f[cluster*nc_agents + j].linear1.weight - x_old1[cluster*nc_agents + j] + gamma*y1[cluster*nc_agents + j])
                    #     z_r2 += R[cluster][ic, j]*(f[cluster*nc_agents + j].linear2.weight - x_old2[cluster*nc_agents + j] + gamma*y2[cluster*nc_agents + j])

                    # z_new1[i] += 1/(T*gamma)*(f[i].linear1.weight - x_old1[i] + gamma*y1[i] - z_r1)
                    # z_new2[i] += 1/(T*gamma)*(f[i].linear2.weight - x_old2[i] + gamma*y2[i] - z_r2)
                for i in range(n_agents):
                    cluster = i//nc_agents
                    ic = i % nc_agents
                    x_new[i] = update_xnew(f, R, N_in, i, nc_agents, x_new[i])
                    # x1_new[i] = R[cluster][ic, ic]*f[i].linear1.weight
                    # x2_new[i] = R[cluster][ic, ic]*f[i].linear2.weight
                    # for j in N_in[cluster][ic]:
                    #     x1_new[i] += R[cluster][ic, j]*f[cluster*nc_agents + j].linear1.weight
                    #     x2_new[i] += R[cluster][ic, j]*f[cluster*nc_agents + j].linear2.weight
                # pdb.set_trace()
                for i in range(n_agents):
                    f[i].load_state_dict(x_new[i])
                    # f[i].linear1.weight = torch.nn.Parameter(x1_new[i])
                    # f[i].linear2.weight = torch.nn.Parameter(x2_new[i])
            # print("after")
            # print(y[0]['linear1.weight'])
            # pdb.set_trace()


            # pdb.set_trace()
            # print("current K round: ", t)

            # print(data_counter[i])

        # com_list = [random.randint(i*nc_agents, (i+1)*nc_agents-1) for i in range(opt.n_cluster)]
        if control == 3:
            sample = int(sample_num[r])
            # pdb.set_trace()
            try:
                com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), sample) for i in range(opt.n_cluster)]
                com_list = [item for sublist in com_list for item in sublist]
            except:
                pdb.set_trace()
        elif control == 1:
            sample_list = [sample_num_opt for i in range(opt.n_cluster)]
            com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), sample_list[i]) for i in range(opt.n_cluster)]
            com_list = [item for sublist in com_list for item in sublist]
        elif control == 2:
            com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
            com_list = [item for sublist in com_list for item in sublist]
        else:
            com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
            com_list = [item for sublist in com_list for item in sublist]

        with torch.no_grad():
            delta_x = [copy.deepcopy(z_i) for i in range(n_agents)]
            # delta_x2 = [0 for i in range(n_agents)]
            for i in range(n_agents):
                for name in f[i].state_dict():
                    delta_x[i][name] = f[i].state_dict()[name] - x_init[i][name] + T*gamma*y[i][name]
                # delta_x1[i] = f[i].linear1.weight - x_init1[i]
                # delta_x2[i] = f[i].linear2.weight - x_init2[i]
                # z1[i] = z_new1[i]
                # z2[i] = z_new2[i]
                z[i] = copy.deepcopy(z_new[i])


            # cluster_x1 = [0 for i in range(n_agents//nc_agents)]
            cluster_x = [copy.deepcopy(z_i) for i in range(n_agents//nc_agents)]
            for i in range(n_agents):
                cluster = i//nc_agents
                ic = i % nc_agents
                if i in com_list:
                    for name in f[i].state_dict():
                        if control == 1:
                            cluster_x[cluster][name] += 1/sample_list[cluster]*delta_x[i][name]
                        elif control == 3:
                            cluster_x[cluster][name] += 1/sample*delta_x[i][name]
                        else:
                            cluster_x[cluster][name] += 1/opt.sample_num*delta_x[i][name]
                    # cluster_x1[cluster] += 1/opt.sample_num*delta_x1[i]
                    # cluster_x2[cluster] += 1/opt.sample_num*delta_x2[i]

        # pdb.set_trace()
        '''aggregate'''
        # x_cen1_delta = 0
        x_cen_delta = copy.deepcopy(z_i)
        # x_cen = copy.deepcopy(z_i)
        with torch.no_grad():
            for n_c in com_list:
                for name in f[n_c].state_dict():
                    if control == 1:
                        x_cen_delta[name] += 1/(opt.n_cluster*sample_list[cluster])*(delta_x[n_c][name])
                    else:
                        x_cen_delta[name] += 1/len(com_list)*(delta_x[n_c][name])
            '''get Gamma_t'''
            if control == 3 and (r+1) % opt.update_rounds == 0:
                Gt = 0
                for n_c in com_list:
                    name_count = 0
                    for name in f[n_c].state_dict():
                        Gt += 1/len(com_list)*torch.norm(delta_x[n_c][name]-x_cen_delta[name])**2
                        name_count += 1
                Gt /= name_count
            
            for name in global_f.state_dict():
                x_cen[name] += x_cen_delta[name]
            if control == 3 and (r+1) % opt.update_rounds == 0:
                '''get Yt'''
                Yt = 0
                for i in range(n_agents//nc_agents):
                    old_cluster_y[i] = copy.deepcopy(cluster_y[i])
                    name_count = 0
                    for name in f[i].state_dict():
                        cluster_y[i][name] = 1/(T*gamma)*(cluster_x[i][name] - x_cen_delta[name])
                        Yt += torch.norm(cluster_y[i][name] - old_cluster_y[i][name])**2
                        name_count += 1
                Yt /= (name_count*(n_agents//nc_agents))
                # pdb.set_trace()
            else:
                for i in range(n_agents//nc_agents):
                    for name in f[i].state_dict():
                        # cluster_y[i][name] = cluster_y[i][name] + 1/(T*gamma)*(cluster_x[i][name] - x_cen_delta[name])
                        cluster_y[i][name] = 1/(T*gamma)*(cluster_x[i][name] - x_cen_delta[name])

                # cluster_y1[i] = cluster_y1[i] + 1/(T*gamma)*(cluster_x1[i] - x_cen1_delta)
                # cluster_y2[i] = cluster_y2[i] + 1/(T*gamma)*(cluster_x2[i] - x_cen2_delta)
            
            '''dynamic control'''
            if control == 3:
                if (r+1) % opt.update_rounds == 0:
                    '''estimate lyapunov function'''
                    BT = opt.update_rounds
                    BT = 1
                    try:
                        Ht = 1/(r+1) + lam[0]**2*(K_list[r]**3*gamma**3/p[r]**2*Yt + K_list[r]*gamma/p[r]*Gt)
                    except:
                        pdb.set_trace()
                    sample_list = []
                    sample_val_list = []
                    for K in range(1, 50):
                        # for i in range(opt.n_cluster):
                        val_list = []
                        for j in range(nc_agents):
                            beta = (nc_agents - (j+1))/nc_agents
                            try:
                                VAL = Ht/((1 - beta**2)**4*BT)*lam[0] + (lam[1]*Ht/(K*BT))**(1/2) + (Ht*lam[1]/(K*(1 - beta**2)**2*BT))**(2/3) + lam[2]*(1 - beta)*sum(energy_list) + lam[2]*K*opt.delta*sum(energy_list)
                            except: 
                                pdb.set_trace()
                            val_list.append(VAL.cpu().numpy())
                        # print(val_list)
                        # pdb.set_trace()
                        sample_list.append(np.argmin(val_list)+1)
                        sample_val_list.append(np.min(val_list))
                    # print('optimal K: %d' % (np.argmin(sample_val_list)+1))
                    K_opt = np.argmin(sample_val_list)+1
                    # print('optimal sample number: %d' % sample_list[np.argmin(sample_val_list)])
                    sample_num_opt = sample_list[np.argmin(sample_val_list)]

                    sample_num[r+1] = sample_num_opt
                    T = K_list[r+1] = K_opt
                    p[r+1] = 1 - ((nc_agents - sample_num_opt)/nc_agents)**2
                    # pdb.set_trace()
                else:
                    sample_num[r+1] = sample_num[r]
                    K_list[r+1] = T
                    p[r+1] = p[r]
            else:
                sample_num[r+1] = sample_num[r]

            '''broadcast'''
            for n_c in com_list:
                cluster = n_c//nc_agents
                ic = n_c % nc_agents

                # y1[n_c] = cluster_y1[cluster]
                # y2[n_c] = cluster_y2[cluster]
                y[n_c] = copy.deepcopy(cluster_y[cluster])

                '''correction term update'''
                # f[n_c].linear1.weight = torch.nn.Parameter(x_cen1)
                # f[n_c].linear2.weight = torch.nn.Parameter(x_cen2)
                f[n_c].load_state_dict(x_cen)
                # pdb.set_trace()

        iters += 1
        com_iter += 1
        com_count += 1
        if control == 1:
            total = 0
            for i in range(opt.n_cluster):
                # total += ((sample_list[i]/nc_agents)**2*energy_list[i])/opt.n_cluster
                total += ((sample_list[i]/nc_agents)*energy_list[i])/opt.n_cluster

            energy_count += total
        elif control == 3:
            total = 0
            for i in range(opt.n_cluster):
                # total += ((sample_list[i]/nc_agents)**2*energy_list[i])/opt.n_cluster
                total += ((sample_num[r]/nc_agents)*energy_list[i])/opt.n_cluster
                total += K_list[r]*opt.delta*energy_list[i]/opt.n_cluster

            energy_count += total
        elif control == 2:
            energy_count += (opt.sample_num/nc_agents)*np.mean(energy_list) + T*opt.delta*np.mean(energy_list)

        # print("[%d agents]: " % n_agents, '\tIters:', iters, '\t\t\t\r')
        # pdb.set_trace()
        if iters % 10 == 0:

            # pdb.set_trace()
            '''compute loss'''
            # x_cen = copy.deepcopy(z_i)
            # for i in range(n_agents):
            #     for name in global_f.state_dict():
            #         x_cen[name] += 1/n_agents*f[i].state_dict()[name]

            # global_f.linear1.weight = torch.nn.Parameter(x_cen1)
            # global_f.linear2.weight = torch.nn.Parameter(x_cen2)
            global_f.load_state_dict(x_cen)
            result = []
            test_counter = 0
            with torch.no_grad():
                while (test_counter + opt.batch <= len(t_input)):
                    test_data = t_input[test_counter: test_counter+opt.batch]
                    if opt.model == 'FNN':
                        test_data = test_data.flatten(start_dim=1)
                    output = global_f(test_data)
                    result.append(output)
                    test_counter += opt.batch
            result = torch.cat(result, dim=0)
            # pdb.set_trace()
            loss = lossfn(result, t_labels[:len(result)])
            t1 = torch.argmax(F.softmax(result, dim=1), dim = 1)
            acc= 0
            for i in range(len(t1)):
                if t1[i] == t_labels[i]:
                    acc += 1
            acc_list.append(acc/len(t1))
            iter_list.append(iters)
            error_list.append(loss.item())
            energy_cost_list.append(energy_count)
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % 100 == 0:
            # print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r')
            print("[%d,%d][%d agents]: " % (T, sample_num[r+1], n_agents), '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], "\tAcc: %.2f" % (100*acc_list[-1]), '\t\t\t\t\r', end = '')

            # pdb.set_trace()
    print('\n', end = '')
    if control >= 1:
        return iter_list, error_list, acc_list, energy_cost_list
    return iter_list, error_list, acc_list


def SCAFFOLD_NN(input, labels, t_input, t_labels, R, C, N_in, N_out, step_c, T=1, n_agents=10, control= 0, opt=None):
    N = input.shape[0]
    dim = input.shape[-1]
    device = torch.device('cuda:{}'.format(opt.gpu_id))
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    label_list = list(torch.split(labels.squeeze(), n, dim=0))
    nc_agents = n_agents//opt.n_cluster

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    # transform_train = nn.Identity()

    # pdb.set_trace()
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
        label_list[-2] = torch.cat((label_list[-2], label_list[-1]), dim=0)
        label_list.pop(-1)
    
    # c = list(zip(input_list, label_list))
    # random.shuffle(c)
    # input_list, label_list = zip(*c)

    if opt.model == 'FNN':
        f = [twoLAYER_NN(dim, opt) for i in range(n_agents)]
        global_f = twoLAYER_NN(dim, opt)
    elif opt.model == 'CNN2':
        f = [ModelCNN(opt) for i in range(n_agents)]
        global_f = ModelCNN(opt)
    elif opt.model == 'TOMCNN':
        f = [TOMCNN(opt) for i in range(n_agents)]
        global_f = TOMCNN(opt)
    else:
        f = [Res_NN(dim, opt) for i in range(n_agents)]
        global_f = Res_NN(dim, opt)
    # pdb.set_trace()

    '''initialization of z, y'''
    z_i = OrderedDict()
    # y_i = {}
    for name in f[0].state_dict():
        z_i[name] = torch.zeros(f[0].state_dict()[name].shape).to(device)
        # y_i[name] = torch.zeros(param.shape).to(device)

    '''initilization of all agents'''
    center_f_dict = global_f.state_dict()
    x_cen = copy.deepcopy(center_f_dict)
    c_cen = copy.deepcopy(z_i)

    # x_cen = copy.deepcopy(z_i)
    for i in range(n_agents):
        f[i].load_state_dict(center_f_dict)
    c = [copy.deepcopy(z_i) for i in range(n_agents)]
    c_new = [copy.deepcopy(z_i) for i in range(n_agents)]

    # z = [copy.deepcopy(z_i) for i in range(n_agents)]
    # z_new = [copy.deepcopy(z_i) for i in range(n_agents)]
    # x_init = [copy.deepcopy(z_i) for i in range(n_agents)]
    # x_old = [copy.deepcopy(z_i) for i in range(n_agents)]
    # x_new = [copy.deepcopy(z_i) for i in range(n_agents)]
    # cluster_y = [copy.deepcopy(z_i) for i in range(n_agents//nc_agents)]

    # x1_new = [0 for i in range(n_agents)]
    # x2_new = [0 for i in range(n_agents)]

    # y1 = [0 for i in range(n_agents)]
    # y2 = [0 for i in range(n_agents)]

    com_count = 0
    gamma = step_c

    error_list = []
    acc_list = []
    iter_list = []

    lossfn = nn.CrossEntropyLoss()

    com_iter = 0
    iters = 0

    '''evaluate'''
    result = []
    test_counter = 0
    with torch.no_grad():
        while (test_counter + opt.batch <= len(t_input)):
            test_data = t_input[test_counter: test_counter+opt.batch]
            if opt.model == 'FNN':
                test_data = test_data.flatten(start_dim=1)
            output = global_f(test_data)
            result.append(output)
            test_counter += opt.batch
    result = torch.cat(result, dim=0)
    # pdb.set_trace()
    loss = lossfn(result, t_labels[:len(result)])
    t1 = torch.argmax(F.softmax(result, dim=1), dim=1)
    acc = 0
    for i in range(len(t1)):
        if t1[i] == t_labels[i]:
            acc += 1
    acc_list.append(acc/len(t1))
    # pdb.set_trace()
    iter_list.append(iters)
    error_list.append(loss.item())
    print("[SCAFFOLD][%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], '\t\t\t\t\r', end = '')

    x_old1 = [0 for i in range(n_agents)]
    # x_old2 = [0 for i in range(n_agents)]
    # z1 = [0 for i in range(n_agents)]
    # z2 = [0 for i in range(n_agents)]
    # z_new1 = [0 for i in range(n_agents)]
    # z_new2 = [0 for i in range(n_agents)]
    # cluster_y1 = [0 for i in range(n_agents)]
    # cluster_y2 = [0 for i in range(n_agents)]

    # x_init1 = [0 for i in range(n_agents)]
    # x_init2 = [0 for i in range(n_agents)]
    # x_cen1 = 0
    # x_cen2 = 0

    # while (com_count < opt.c_rounds):
    # T = int(opt.p_inv)
    data_counter = [0 for i in range(n_agents)]
    energy_count = 0
    energy_cost_list = []

    for r in range(int(opt.c_rounds)):
        # for i in range(n_agents):
        #     # z_new1[i] = z1[i]
        #     # z_new2[i] = z2[i]
        #     z_new[i] = copy.deepcopy(z[i])
        #     # x_init1[i] = f[i].linear1.weight
        #     # x_init2[i] = f[i].linear2.weight
        #     x_init[i] = copy.deepcopy(f[i].state_dict())
        com_list = [random.sample(range(i*nc_agents, (i+1)*nc_agents), opt.sample_num) for i in range(opt.n_cluster)]
        com_list = [item for sublist in com_list for item in sublist]
        # pdb.set_trace()
        for i in com_list:
            f[i].load_state_dict(x_cen)
        # pdb.set_trace()

        for t in range(T):
            for i in com_list:
                # print(i)
                if data_counter[i] + opt.batch <= len(input_list[i]):
                    i_input = input_list[i][data_counter[i]:data_counter[i] + opt.batch]
                    i_label = label_list[i][data_counter[i]:data_counter[i] + opt.batch]
                    data_counter[i] = (data_counter[i] + opt.batch) % len(input_list[i])
                else:
                    i_input1 = input_list[i][data_counter[i]:-1]
                    i_label1 = label_list[i][data_counter[i]:-1]
                    length = opt.batch - i_input1.shape[0]
                    i_input = torch.cat([input_list[i][0: length], i_input1], dim=0)
                    i_label = torch.cat([label_list[i][0: length], i_label1], dim=0)
                    data_counter[i] = (data_counter[i] + opt.batch + 1) % len(input_list[i])

                if opt.dataset != 'MNIST':
                    i_input = transform_train(i_input)
                if opt.model == 'FNN':
                    i_input = i_input.flatten(start_dim=1)
                f[i].zero_grad()
                output = f[i](i_input)
                loss = lossfn(output, i_label)
                loss.backward()

                # if i == 0:
                #     print(f[0].linear2.weight)
                with torch.no_grad():
                    # x_old1[i] = f[i].linear1.weight
                    # x_old2[i] = f[i].linear2.weight
                    new_param = OrderedDict()
                    t1 = f[i].state_dict()
                    for name in f[i].state_dict():
                        if ('num_batches_tracked' in name or 'running_var' in name or 'running_mean' in name):
                            del t1[name]
                    grad_dict = {k: v.grad for k, v in zip(t1, f[i].parameters())}
                    black_list = ['num_batches_tracked', 'running_var', 'running_mean']
                    for name in f[i].state_dict():
                        # try:
                        if not ('num_batches_tracked' in name or 'running_var' in name or 'running_mean' in name):
                            new_param[name] = f[i].state_dict()[name] - gamma*(c_cen[name] - c[i][name] + grad_dict[name])
                        else:
                            new_param[name] = f[i].state_dict()[name]
                    f[i].load_state_dict(new_param)

                    # cluster_x2[cluster] += 1/opt.sample_num*delta_x2[i]
        # pdb.set_trace()
        '''aggregate'''
        # x_cen1_delta = 0
        x_cen_delta = copy.deepcopy(z_i)
        c_cen_delta = copy.deepcopy(z_i)
        # x_cen = copy.deepcopy(z_i)
        # pdb.set_trace()

        with torch.no_grad():
            for i in com_list:
                for name in f[i].state_dict():
                    c_new[i][name] = c[i][name] - c_cen[name] + 1/(T*gamma)*(x_cen[name] - f[i].state_dict()[name])
            for i in com_list:
                for name in f[i].state_dict():
                    # x_cen_delta[name] += 1/len(com_list)*(f[i].state_dict()[name] - x_cen[name])
                    c_cen_delta[name] += 1/len(com_list)*(c_new[i][name] - c[i][name])
            for i in range(n_agents):
                for name in f[i].state_dict():
                    x_cen_delta[name] += 1/n_agents*(f[i].state_dict()[name] - x_cen[name])

            for i in com_list:
                c[i] = copy.deepcopy(c_new[i])
            for name in global_f.state_dict():
                x_cen[name] = x_cen[name] + x_cen_delta[name]
                c_cen[name] = c_cen[name] + len(com_list)/n_agents*c_cen_delta[name]
        # pdb.set_trace()


        iters += 1
        com_iter += 1
        com_count += 1


        # print("[%d agents]: " % n_agents, '\tIters:', iters, '\t\t\t\r')
        # pdb.set_trace()
        if iters % 10 == 0:

            # pdb.set_trace()
            '''compute loss'''

            global_f.load_state_dict(x_cen)
            result = []
            test_counter = 0
            with torch.no_grad():
                while (test_counter + opt.batch <= len(t_input)):
                    test_data = t_input[test_counter: test_counter+opt.batch]
                    if opt.model == 'FNN':
                        test_data = test_data.flatten(start_dim=1)
                    output = global_f(test_data)
                    result.append(output)
                    test_counter += opt.batch
            result = torch.cat(result, dim=0)
            # pdb.set_trace()
            loss = lossfn(result, t_labels[:len(result)])
            t1 = torch.argmax(F.softmax(result, dim=1), dim=1)
            acc = 0
            for i in range(len(t1)):
                if t1[i] == t_labels[i]:
                    acc += 1
            acc_list.append(acc/len(t1))
            iter_list.append(iters)
            error_list.append(loss.item())
            energy_cost_list.append(energy_count)
        # error_list.append((torch.norm(torch.stack(x_list, dim=1) - x_opt.unsqueeze(1))/n_agents).item())

        if iters % 100 == 0:
            print("[%d agents]: " % n_agents, '\tIters:', iters, '\tLoss: %.2E' % error_list[-1], "\tAcc: %.2f" % (100*acc_list[-1]), '\t\t\t\t\r', end = '')
    print('\n', end = '')
    return iter_list, error_list, acc_list
