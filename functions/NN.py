import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import Type, Any, Callable, Union, List, Optional
# from .caps_pool import dynamic_pool
import pdb



class twoLAYER_NN(nn.Module):
    def __init__(self, dim, opt):
        super(twoLAYER_NN, self).__init__()
        device = torch.device('cuda:{}'.format(opt.gpu_id))
        if opt.dataset == 'MNIST' or opt.dataset == 'CIFAR10':
            n_labels = 10
        else:
            n_labels = 100

        if opt.dataset == 'MNIST':
            dim1 = dim**2
        else:
            dim1 = dim**2*3
        if opt.dataset == 'MNIST':
            dim2 = dim1//4
        else:
            dim2 = dim1//2
        self.linear1 = nn.Linear(dim1, dim2).to(device)
        # if opt.dataset != 'MNIST':
        #     self.linear1 = nn.Sequential(
        #         nn.Linear(dim1, dim2),
        #         nn.ReLU(),
        #         nn.Linear(dim2, dim2)).to(device)
        self.relu = nn.ReLU()
        
        self.linear2 = nn.Linear(dim2, n_labels).to(device)

    def forward(self, data):
        x1 = self.linear1(data)
        x2 = self.relu(x1)
        x3 = self.linear2(x2)
        # y = F.log_softmax(x3, dim = 1)

        return x3
    

class Res_NN(nn.Module):
    def __init__(self, dim, opt):
        super(Res_NN, self).__init__()
        device = torch.device('cuda:{}'.format(opt.gpu_id))
        self.dim = dim
        if opt.dataset == "MNIST":
            in_channel = 1
        else:
            in_channel = 3
        channel_list = [8, 16]
        norm_layer = nn.BatchNorm2d
        # norm_layer = nn.Identity
        # norm_layer = nn.InstanceNorm2d


        self.resblock1 = BasicBlock(in_channel, channel_list[0], stride=2, norm_layer=norm_layer).to(device)
        self.resblock2 = BasicBlock(channel_list[0], channel_list[1], stride=2, norm_layer=norm_layer).to(device)
        # self.resblock1 = BasicBlock(in_channel, channel_list[0], stride=2, norm_layer=nn.Identity).to(device)
        # self.resblock2 = BasicBlock(channel_list[0], channel_list[1], stride=2, norm_layer=nn.Identity).to(device)
        self.relu = nn.ReLU()

        if opt.dataset == 'MNIST' or opt.dataset == 'CIFAR10':
            n_labels = 10
        else:
            n_labels = 100

        self.linear = nn.Linear((dim//4)**2*channel_list[-1], n_labels).to(device)

    def forward(self, data):
        x1 = self.resblock1(data)
        x2 = self.relu(x1)
        x3 = self.resblock2(x2)
        x4 = self.relu(x3)
        x5 = self.linear(x4.flatten(1))
        # pdb.set_trace()
        # y = F.log_softmax(x3, dim = 1)
        return x5
    

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        # spatial: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        # last_block=False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum = 0, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum = 0, track_running_stats=False)
        self.downsample = nn.Sequential(
            conv1x1(inplanes, planes, stride),
            norm_layer(planes, momentum = 0, track_running_stats=False)
            )
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.m1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ModelCNN(nn.Module):
    def __init__(self, opt):
        super(ModelCNN, self).__init__()
        device = torch.device('cuda:{}'.format(opt.gpu_id))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ).to(device)

        if opt.dataset == 'CIFAR10':
            self.fc1 = nn.Sequential(
                nn.Linear(8 * 8 * 32, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
            ).to(device)
            self.fc2 = nn.Linear(64, 10).to(device)
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(8 * 8 * 32, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            ).to(device)
            self.fc2 = nn.Linear(256, 100).to(device)
        # Use Kaiming initialization for layers with ReLU activation
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.conv.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self, x):
        conv_ = self.conv(x)
        fc_ = conv_.view(-1, 8 * 8 * 32)
        fc1_ = self.fc1(fc_)
        output = self.fc2(fc1_)
        return output


class TOMCNN(nn.Module):
    def __init__(self, opt):
        super(TOMCNN, self).__init__()
        device = torch.device('cuda:{}'.format(opt.gpu_id))

        if opt.dataset == 'CIFAR10' or opt.dataset == 'MNIST':
            self.n_cls = 10
        else:
            self.n_cls = 100
        # Use Kaiming initialization for layers with ReLU activation
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5).to(device)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5).to(device)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2).to(device)
        self.fc1 = nn.Linear(64*5*5, 384).to(device)
        self.fc2 = nn.Linear(384, 192).to(device)
        self.fc3 = nn.Linear(192, self.n_cls).to(device)
        @torch.no_grad()
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x) 
        return output
