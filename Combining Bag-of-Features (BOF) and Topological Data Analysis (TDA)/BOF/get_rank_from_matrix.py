import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import random

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = alpha + random.uniform(alpha/100, alpha/10)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Effective_Ranks:
    # 输入的是一个矩阵，得到的是这个矩阵的ranks，具体的讲就是
    #  R0,r0,rk,Rk
    def __init__(self, x):
        self.x = x
        self.cov_operator = None
        self.eigenvalues = None
        self.get_cov_operator()
        self.get_eigenvalues()

        self.rk = self.get_rk()
        self.r0 = self.rk[0]
        self.rk_max_value = max(self.rk)
        self.rk_max_index = self.rk.index(self.rk_max_value)

        self.Rk = self.get_Rk()
        self.R0 = self.Rk[0]
        self.Rk_value_max_rk_index = self.Rk[self.rk_max_index]

    def get_cov_operator(self):
       self.cov_operator = np.cov(self.x, rowvar=False)
      
    def get_eigenvalues(self):
       
       self.eigenvalues = np.sort(np.linalg.eigvalsh(self.cov_operator))[::-1]
       
    def get_rk(self):
       r_k = [self.eigenvalues[k::1].sum() / (self.eigenvalues[k]+10**-10) for k in range(len(self.eigenvalues)-1)]
       return r_k
    
    def get_Rk(self):
       R_k = [(self.eigenvalues[k::1].sum())**2 / np.sum(self.eigenvalues[k::1]**2) for k in range(len(self.eigenvalues))]
       return R_k
    
