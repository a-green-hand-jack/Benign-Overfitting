import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np



# 定义一个测试函数用来确定在Colab上成功加载文件夹
def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from get_rank_0904.py!')

# 在一个Class 中实现对有效秩的计算的封装

class Effective_Ranks:
    '''这个类是为了计算x的有效秩,这里的x是一个矩阵，矩阵的行数是样本数，列数是每一个样本的特征数，下面是一个案例:
      >>>x = np.random.rank(3, 5)
      >>>effective_ranks = Effective_Ranks(x)
      >>>print("得到有效秩:\nr_k:{0}\nR_k:{1}".format(effective_ranks.get_rk(), effective_ranks.get_Rk()))
      >>>得到有效秩:
         r_k:[4.900916823810114, 1329526124078698.8, 7.503499842976844, -8.224181034995697]
         R_k:[1.4810853111404272, 1.0000000000000018, 1.2587441340275045, 0.7856972580607169, 1.0]
   '''
    
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
      device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  ):
        self.train_rk = None
        self.train_Rk = None
        self.train_loader = train_dataloader
        self.device = device
        if data_train_matrix == None: 
            self.convert_to_matrix()
            self.convert_to_rank()
        else:
            self.train_matrix = data_train_matrix
            self.convert_to_rank()

    def convert_to_matrix(self):
        train_vectors = []
        for images, labels in self.train_loader:
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