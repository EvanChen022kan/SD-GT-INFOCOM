import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
# from .gradfn import func
# from .utils import get_mse_fstar, build_asy_list
# from .NN import twoLAYER_NN, Res_NN, ModelCNN, TOMCNN
from torchvision import transforms
import random
import scipy.io as sio
import copy
import os
from collections import OrderedDict
# from .GT_func import *
import pdb


def save_config(data, network, opt, model, mode):
    dataset = opt.dataset
    if opt.setting1:
        exp = 'setting1'
    elif opt.setting2:
        exp = 'setting2'
    elif opt.setting3:
        exp = 'setting3'


    # [global_f, f, y, z, x_cen, z_new, x_init, x_old, x_new, cluster_y, old_cluster_y]
    folder_name = os.path.join("checkpoints", '%s-%s-%s' % (str(opt.config_id), dataset, exp))
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    if mode == 'params':
        filename = '%d-%d-%d-%s-params.pth' % (network[0], network[1], network[2], model)
        pathname = os.path.join(folder_name, filename)
        if model == 'SDGT':
            param_dict = {}
            param_dict['global_f'] = data[0].state_dict()
            param_dict['f'] = [data[1][i].state_dict() for i in range(len(data[1]))]
            param_dict['y'] = data[2]
            param_dict['z'] = data[3]
            param_dict['x_cen'] = data[4]
            param_dict['z_new'] = data[5]
            param_dict['x_init'] = data[6]
            param_dict['x_old'] = data[7]
            param_dict['x_new'] = data[8]
            param_dict['cluster_y'] = data[9]
            param_dict['old_cluster_y'] = data[10]
            # pdb.set_trace()
            torch.save(param_dict, pathname)
        elif model == 'FedAvg':
            param_dict = {}
            param_dict['global_f'] = data[0].state_dict()
            param_dict['f'] = [data[1][i].state_dict() for i in range(len(data[1]))]
            param_dict['x_cen'] = data[2]
            param_dict['x_new'] = data[3]
            torch.save(param_dict, pathname)
        elif model == 'SCAFFOLD':
            param_dict = {}
            param_dict['global_f'] = data[0].state_dict()
            param_dict['f'] = [data[1][i].state_dict() for i in range(len(data[1]))]
            param_dict['x_cen'] = data[2]
            param_dict['c_cen'] = data[3]
            param_dict['c_new'] = data[4]
            param_dict['c'] = data[5]
            torch.save(param_dict, pathname)


    elif mode == 'network':
        filename = '%d-%d-%d-network.pth' % (network[0], network[1], network[2])
        pathname = os.path.join(folder_name, filename)
        torch.save(data, pathname)
    elif mode == 'plot':
        filename = '%d-%d-%d-%s-plot.pth' % (network[0], network[1], network[2], model)
        pathname = os.path.join(folder_name, filename)
        torch.save(data, pathname)




def load_config(data, network, opt, model, mode):
    dataset = opt.dataset
    if opt.setting1:
        exp = 'setting1'
    elif opt.setting2:
        exp = 'setting2'
    elif opt.setting3:
        exp = 'setting3'
    folder_name = os.path.join("checkpoints", '%s-%s-%s' % (str(opt.config_id), dataset, exp))
    if mode == 'params':
        if model == 'SDGT':
            filename = '%d-%d-%d-%s-params.pth' % (network[0], network[1], network[2], model)
            pathname = os.path.join(folder_name, filename)
            # pdb.set_trace()
            if os.path.exists(pathname):
                print('continue training from %s' % pathname)
                t1 = []
                new_data = torch.load(pathname)
                data[0].load_state_dict(new_data['global_f'])
                for i in range(len(new_data['f'])):
                    data[1][i].load_state_dict(new_data['f'][i]) 
                t1.append(new_data['y'])
                t1.append(new_data['z'] )
                t1.append(new_data['x_cen'])
                t1.append(new_data['z_new'] )
                t1.append(new_data['x_init'] )
                t1.append(new_data['x_old'] )
                t1.append(new_data['x_new'])
                t1.append(new_data['cluster_y'] )
                t1.append(new_data['old_cluster_y'] )
                return t1[0], t1[1],t1[2],t1[3],t1[4],t1[5],t1[6],t1[7],t1[8]
            else:
                t1 = data
                return t1[2],t1[3],t1[4],t1[5],t1[6],t1[7],t1[8],t1[9], t1[10]
        elif model == 'FedAvg':
            filename = '%d-%d-%d-%s-params.pth' % (network[0], network[1], network[2], model)
            pathname = os.path.join(folder_name, filename)
            if os.path.exists(pathname):
                print('continue training from %s' % pathname)

                t1 = []
                new_data = torch.load(pathname)
                data[0].load_state_dict(new_data['global_f'])
                for i in range(len(new_data['f'])):
                    data[1][i].load_state_dict(new_data['f'][i]) 
                # t1.append(new_data['x_cen'])
                t1.append(new_data['x_new'])
                return t1[0]
            else:
                t1 = data
                return t1[2]
        elif model == 'SCAFFOLD':
            filename = '%d-%d-%d-%s-params.pth' % (network[0], network[1], network[2], model)
            pathname = os.path.join(folder_name, filename)
            if os.path.exists(pathname):
                print('continue training from %s' % pathname)
                t1 = []
                new_data = torch.load(pathname)
                data[0].load_state_dict(new_data['global_f'])
                for i in range(len(new_data['f'])):
                    data[1][i].load_state_dict(new_data['f'][i]) 
                t1.append(new_data['x_cen'])
                t1.append(new_data['c_cen'])
                t1.append(new_data['c_new'])
                t1.append(new_data['c'])
                return t1[0], t1[1],t1[2],t1[3]
            else:
                t1 = data
                return t1[2],t1[3],t1[4],t1[5]
        # pdb.set_trace()
    elif mode == 'network':
        filename = '%d-%d-%d-network.pth' % (network[0], network[1], network[2])
        pathname = os.path.join(folder_name, filename)
        if os.path.exists(pathname):
            data = torch.load(pathname)
            return data[0], data[1],data[2],data[3]
        else:
            return data[0], data[1],data[2],data[3]
    elif mode == 'plot':
        filename = '%d-%d-%d-%s-plot.pth' % (network[0], network[1], network[2], model)
        pathname = os.path.join(folder_name, filename)
        if os.path.exists(pathname):
            data = torch.load(pathname)
            return data[0], data[1],data[2]
        else:
            return data[0],data[1],data[2]
