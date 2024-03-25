import torch
import numpy as np
import pdb



class func():
    def __init__(self, input, labels, lam, opt) -> None:
        self.input = input
        self.labels = labels
        self.lam = lam
        self.loss = opt.loss

    def f(self, x):
        if self.loss == 'mse':
            return msefn(x, self.input, self.labels, self.lam)
        else:
            return fn(x, self.input, self.labels, self.lam)
    def g(self, x):
        if self.loss == 'mse':
            return msegfn(x, self.input, self.labels, self.lam)
        else:
            return gfn(x, self.input, self.labels, self.lam)




def fn(x, input, labels, lam = 0):
    # N = input.shape[0]
    a_x = torch.einsum('nd,d -> n', input, x)
    ba_x = torch.einsum('n,n -> n', a_x, labels)
    loss = torch.mean(torch.log(1 + torch.exp(-ba_x))) + lam/2*torch.norm(x)**2
    return loss


def gfn(x, input, labels, lam = 0):
    a_x = torch.einsum('nd,d -> n', input, x)
    ba_x = torch.einsum('n,n -> n', a_x, labels)
    grad = torch.mean(-torch.einsum('nd,n -> nd', input, labels/(1 + torch.exp(ba_x))), dim=0) + lam*x
    
    return grad


def msefn(x, input, labels, lam=0):
    # N = input.shape[0]
    f = torch.norm(input @ x - labels)**2 + lam/2*torch.norm(x)**2
    return f


def msegfn(x, input, labels, lam=0):
    # N = input.shape[0]
    gf = 2*(input.transpose(1,0) @ (input @ x - labels)) + lam*x
    # pdb.set_trace()
    return gf

def f(x, input, labels, lam):
    N = input.shape[0]
    res = 0
    for n in range(N):
        res = res + np.log(1 + np.exp(-labels[n]*np.matmul(input[n], x)))
        # if (1 + np.exp(-labels[n]*np.matmul(input[n], x))) > 10:
        #     pdb.set_trace()
        # pdb.set_trace()
    res /= N
    res += lam/2*np.linalg.norm(x)
    return res


def f_der(x, input, labels, lam):
    N = input.shape[0]
    dim = x.shape[0]
    der = np.zeros(dim)
    for n in range(N):
        der = der - labels[n]*input[n]/(1 + np.exp(labels[n]*np.matmul(input[n], x)))
    der /= N
    der += lam*x
    return der
