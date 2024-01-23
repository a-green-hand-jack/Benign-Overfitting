import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import random
import torch
from typing import List, Union

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
    def __init__(self, x: Union[List[List[float]], List[List[int]]]):
        """
        Calculates ranks of a matrix.

        Args:
        - x: Input matrix (list of lists of floats or integers).
        """
        self.x = x
        self.cov_operator = None  # Covariance operator of the matrix
        self.eigenvalues = None  # Eigenvalues of the covariance operator
        self.get_cov_operator()
        self.get_eigenvalues()

        self.rk = self.get_rk()
        self.r0 = self.rk[0]
        self.rk_max_value = max(self.rk)
        self.rk_max_index = self.rk.index(self.rk_max_value)

        self.Rk = self.get_Rk()
        self.R0 = self.Rk[0]
        self.Rk_value_max_rk_index = self.Rk[self.rk_max_index]

    def get_cov_operator(self) -> None:
        """
        Computes the covariance operator of the input matrix.
        """
        self.cov_operator = np.cov(self.x, rowvar=False)
      
    def get_eigenvalues(self) -> None:
        """
        Computes the eigenvalues of the covariance operator.
        """
        eigenvalues = np.linalg.eigvalsh(self.cov_operator)
        self.eigenvalues = np.sort(eigenvalues)[::-1]
       
    def get_rk(self) -> List[float]:
        """
        Computes r_k values.
        """
        r_k = [
            self.eigenvalues[k::1].sum() / (self.eigenvalues[k] + 10 ** -10)
            for k in range(len(self.eigenvalues) - 1)
        ]
        return r_k
    
    def get_Rk(self) -> List[float]:
        """
        Computes R_k values.
        """
        R_k = [
            (self.eigenvalues[k::1].sum()) ** 2 / np.sum(self.eigenvalues[k::1] ** 2)
            for k in range(len(self.eigenvalues))
        ]
        return R_k

    
    

class Effective_Ranks_GPU:
    def __init__(self, x: torch.Tensor):
        """
        Calculates ranks of a matrix on GPU.

        Args:
        - x: Input matrix as a PyTorch tensor (should be on GPU).
        """
        self.x = x
        self.cov_operator = None  # Covariance operator of the matrix
        self.eigenvalues = None  # Eigenvalues of the covariance operator
        self.get_cov_operator()
        self.get_eigenvalues()

        self.rk = self.get_rk()
        self.rk = self.get_rk()
        self.r0 = self.rk[0]
        self.rk_max_value = max(self.rk)
        self.rk_max_index = self.rk.index(self.rk_max_value)

        self.Rk = self.get_Rk()
        self.R0 = self.Rk[0]
        self.Rk_value_max_rk_index = self.Rk[self.rk_max_index]

    def get_cov_operator(self) -> None:
        """
        计算输入矩阵的协方差算子。
        
        协方差算子定义为：
        协方差算子 = \frac{X^T \cdot X}{n}
        其中，X^T 表示矩阵 X 的转置，n 是数据点的数量。
        """
        self.cov_operator = torch.matmul(self.x.t(), self.x) / self.x.size(0)

    def get_eigenvalues(self) -> None:
        """
        计算协方差算子的特征值。
        
        通过 torch.linalg.eigh 函数计算协方差算子的特征值。
        特征值满足以下方程：
        C v = \lambda v
        其中，C 是协方差算子，v 是特征向量，\lambda 表示特征值。

        这两行代码使用 PyTorch 中的函数 `torch.linalg.eigh()` 计算了给定协方差算子的特征值，并将结果按照降序排列后赋值给了 `self.eigenvalues`。

        具体来说：
        - `torch.linalg.eigh()` 函数用于计算 Hermitian（或实对称）矩阵的特征值和特征向量。在这里，`self.cov_operator` 应该是一个实对称矩阵（协方差矩阵），该函数返回特征值和特征向量。
        - 由于只关心特征值，所以使用 `_` 接收了函数返回的特征向量，而特征值则被赋值给了 `eigenvalues` 变量。
        - `self.eigenvalues = eigenvalues.flip(0)` 通过 `flip(0)` 将特征值按照维度 0（即第一个维度，通常是行）进行翻转，实现了将特征值按降序排列的操作。最终结果被赋值给了 `self.eigenvalues` 属性。
        """
        eigenvalues, _ = torch.linalg.eigh(self.cov_operator)
        self.eigenvalues = eigenvalues.flip(0)

    def get_rk(self) -> List[float]:
        """
        Computes r_k values.
        """
        r_k = [
            self.eigenvalues[k:].sum() / (self.eigenvalues[k] + 1e-10)
            for k in range(len(self.eigenvalues) - 1)
        ]
        return r_k

    def get_Rk(self) -> List[float]:
        """
        Computes R_k values.
        """
        R_k = [
            (self.eigenvalues[k:].sum() ** 2) / (self.eigenvalues[k:] ** 2).sum() 
            for k in range(len(self.eigenvalues))
        ]
        return R_k

    
