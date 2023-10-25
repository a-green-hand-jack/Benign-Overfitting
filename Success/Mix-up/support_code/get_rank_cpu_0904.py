import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
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
    
    def __init__(self, x):
      self.x = x
      self.cov_operator = None
      self.eigenvalues = None
      self.get_cov_operator()
      self.get_eigenvalues()
      # 注意类变量和方法不能重名

    def get_cov_operator(self):
       self.cov_operator = np.cov(self.x, rowvar=False)
      #  return cov_operator
    def get_eigenvalues(self):
       
       self.eigenvalues = np.sort(np.linalg.eigvalsh(self.cov_operator))[::-1]
       
    
    def get_rk(self):
       r_k = [self.eigenvalues[k::1].sum() / (self.eigenvalues[k]+10**-10) for k in range(len(self.eigenvalues)-1)]
       return r_k
    
    def get_Rk(self):
       R_k = [(self.eigenvalues[k::1].sum())**2 / np.sum(self.eigenvalues[k::1]**2) for k in range(len(self.eigenvalues))]
       return R_k
    
# 如果只是加载一种数据库，感觉建立一个class的意义不能体现
class get_Effective_Ranks:
    """
    给我一个data_loader,返回一个rank
    """
    def __init__(
      self, 
      data_train_matrix=None, 
      train_dataloader=None, 
      device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
      alpha = None
  ):
        self.train_rk = None
        self.train_Rk = None
        self.train_loader = train_dataloader
        self.device = device
        self.alpha = alpha
        if data_train_matrix == None: 
            self.convert_to_matrix()
            self.convert_to_rank()
        else:
            self.train_matrix = data_train_matrix
            self.convert_to_rank()

    def convert_to_matrix(self):
        train_vectors = []
        for images, labels in self.train_loader:
            images = torch.Tensor(images)
            labels = torch.Tensor(labels)
            images, _, _, _ = mixup_data(images, labels, self.alpha, False)
            # images, _, _ = map(Variable, (images,_, _))
            images = images.view(images.size(0), -1)
            train_vectors.append(images[0])
        self.train_matrix = np.stack(train_vectors)
        # self.train_matrix = torch.tensor(self.train_matrix, dtype=torch.float32).to(self.device)  # 移动到指定设备

    def convert_to_rank(self):
        # self.train_effective_rank = Effective_Ranks(self.train_matrix.to('cpu').numpy())  # 注意要先将 Tensor 移回 CPU
        self.train_effective_rank = Effective_Ranks(self.train_matrix)
        self.train_rk = self.train_effective_rank.get_rk()
        self.train_Rk = self.train_effective_rank.get_Rk()

    def plot_vectors(self):
        plt.plot(self.train_rk, label='rk')
        plt.plot(self.train_Rk, label='Rk')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('rk & Rk in {0}'.format(self.my_transform.transforms[0]))
        plt.legend()
        plt.show()