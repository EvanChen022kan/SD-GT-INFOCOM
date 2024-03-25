from datetime import datetime
import pytz
import os
import pickle
import torch
import random
from .gradfn import msefn
import pdb


def save_img(plt, name, opt):
    current_datetime = datetime.now(pytz.timezone('US/East-Indiana'))
    current_date_time = current_datetime.strftime("%m-%d-%H-%M-%S")
    current_date = current_datetime.strftime("%m-%d")
    dir_name = os.path.join(opt.result_path, current_date)
    filename = name + '-' + current_date_time + '.jpg'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(os.path.join(dir_name, filename), dpi=300, bbox_inches="tight")


def save_plt(plt_list, name, opt):
    current_datetime = datetime.now(pytz.timezone('US/East-Indiana'))
    current_date_time = current_datetime.strftime("%m-%d-%H-%M-%S")
    current_date = current_datetime.strftime("%m-%d")
    dir_name = os.path.join(opt.result_path, current_date)
    filename = name + '-' + current_date_time + '.pth'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(plt_list, os.path.join(dir_name, filename))

def update_fstar(val, x, n_agents):
    fi = open('f_star.pkl', 'rb')
    f_star_dict = pickle.load(fi)
    f_star_dict[n_agents] = (val, x.cpu())
    f = open("f_star.pkl", "wb")
    pickle.dump(f_star_dict, f)
    f.close()


def get_fstar(n_agents):
    fi = open('f_star.pkl', 'rb')
    f_star_dict = pickle.load(fi)
    if f_star_dict.get(n_agents) is None:
        f_star = 100
        x = None
    else:
        (f_star, x) = f_star_dict[n_agents]
    return f_star, x


def del_fstar(n_agents):
    fi = open('f_star.pkl', 'rb')
    f_star_dict = pickle.load(fi)
    f_star_dict.pop(n_agents)
    f = open("f_star.pkl", "wb")
    pickle.dump(f_star_dict, f)
    f.close()

def get_mse_fstar(input_list, label_list, lam):
    N = len(input_list)
    dim = input_list[0].shape[1]
    I = torch.eye(dim).to(input_list[0].device)
    AA_list = []
    Ab_list = []
    for i in range(N):
        # t1 = torch.einsum('na,nb -> ab', input_list[i], input_list[i])
        # t1 += lam*I
        # t2 = torch.einsum('na,b -> a', input_list[i], label_list[i])
        AA = input_list[i].transpose(1, 0) @ input_list[i] + lam/2*I
        AA_list.append(AA)
        Ab_list.append(input_list[i].transpose(1, 0) @ label_list[i])
        # pdb.set_trace()

    AA = torch.sum(torch.stack(AA_list), dim = 0)
    Ab = torch.sum(torch.stack(Ab_list), dim = 0)
    AAinv = torch.linalg.inv(AA)
    x_opt = AAinv @ Ab

    def global_f(x): return torch.mean(torch.stack([msefn(x, input_list[i], label_list[i], lam) for i in range(N)]))
    fstar = global_f(x_opt)
    # x0 = torch.randn(dim, dtype=torch.double).to(fstar.device)
    # x1 = torch.zeros(dim, dtype=torch.double).to(fstar.device)

    return fstar, x_opt

def get_L(input, n_agents, opt):
    import torch
    N = input.shape[0]
    dim = input.shape[1]
    n = N//n_agents
    input_list = list(torch.split(input, n, dim=0))
    if len(input_list) != n_agents:
        input_list[-2] = torch.cat((input_list[-2], input_list[-1]), dim=0)
        input_list.pop(-1)
    L_list = []
    if opt.loss == 'mse':
        for i in range(len(input_list)):
            AA = 2*input_list[i].transpose(1, 0) @ input_list[i]
            L_list.append(torch.max(torch.linalg.eigvals(AA).real))
        return torch.mean(torch.stack(L_list))
    else:
        for i in range(len(input_list)):
            nabla2_f = torch.zeros(dim, dim).to(input.device)
            for j in range(len(input_list[i])):
                nabla2_f = nabla2_f + 1/4*torch.einsum('n,k->nk', input_list[i][j], input_list[i][j])
                # pdb.set_trace()
            nabla2_f /= len(input_list[i])
            L_list.append(torch.norm(nabla2_f))
        return torch.mean(torch.stack(L_list))



def build_asy_list(total_iters, round_max, n_agents):
    output = []
    while(len(output) < total_iters):
        length = random.randint(n_agents, round_max)
        seq = [i for i in range(n_agents)] + [random.randint(0, n_agents-1) for i in range(length - n_agents)]
        random.shuffle(seq)
        output = output + seq
    
    return output[:total_iters]
