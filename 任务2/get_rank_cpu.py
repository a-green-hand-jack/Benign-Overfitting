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
  print('Hello from get_rank.py!')

# 在一个Class 中实现对有效秩的计算的封装

class Effective_Ranks:
    '''这个类是为了计算x的有效秩,这里的x是一个矩阵，矩阵的行数是样本数，列数是每一个样本的特征数，下面是一个案例:
      >>>x = np.random.randn(3, 5)
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
       r_k = [self.eigenvalues[k::1].sum() / self.eigenvalues[k] for k in range(len(self.eigenvalues)-1)]
       return r_k
    
    def get_Rk(self):
       R_k = [(self.eigenvalues[k::1].sum())**2 / np.sum(self.eigenvalues[k::1]**2) for k in range(len(self.eigenvalues))]
       return R_k
    
# 如果只是加载一种数据库，感觉建立一个class的意义不能体现
class get_Effective_Ranks:
    """
    为了使后续的调用更加简单，专门定义了这个类。只要输入对应的数据库的位置就可以得到这个数据库的协变量的协方差矩阵的有效秩。使用方法如下：
    >>> get_cifar10_2 = get_Effective_Ranks(dataset_name='CIFAR10', path_to_dataset_folder='dataset_folder')
    >>> print(get_cifar10_2.train_rk, "\n",get_cifar10_2.train_rk)
    >>> Files already downloaded and verified
    >>> Files already downloaded and verified
    >>> [...]   
    >>> [...]   # 太长了，就不写了
    为了避免重复加载数据，如果已经得到了数据库对应的图片的矩阵；就会跳过加载过程过程，直接计算协方差算子矩阵的有效秩
    --------------------这是一个分割线-------------------------
    后来为了可以实现transform的快速定义,增加了my_transform = transforms.Compose([ToTensor(),])这样个默认参数,默认参数表示不对样本进行任何操作.
    下面给出一个操作的实例:
    # 首先是一个transform的操作list
    >>> my_transforms_list = [
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
        ]
    >>> get_cifar10 = get_Effective_Ranks(dataset_name='CIFAR10', path_to_dataset_folder='dataset_folder')
    >>> rk_1, Rk_2 = get_cifar10.train_rk, get_cifar10.train_Rk
    >>> get_cifar10.plot_vectors()
    # 通过这3步就可以得到没有经过预处理的有效秩的情况
    >>> for i in my_transforms_list:
    >>>     transform = transforms.Compose([i, transforms.ToTensor(),])
    >>>     get_cifar10 = get_Effective_Ranks(dataset_name='CIFAR10', path_to_dataset_folder='dataset_folder', my_transform=transform)
    >>>     rk_1, Rk_2 = get_cifar10.train_rk, get_cifar10.train_Rk
    >>>     get_cifar10.plot_vectors()
    # 通过这个for循环得到了每一种预处理之后的图片的有效秩关于k的变化的曲线图
    """
    def __init__(self, dataset_name, path_to_dataset_folder, val_split=0.1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), data_train_matrix=None, my_transform=transforms.Compose([ToTensor(),])):
        self.device = device  # 添加设备参数
        self.dataset_name = dataset_name
        self.path_to_dataset_folder = path_to_dataset_folder
        self.train_rk = None
        self.train_Rk = None
        # self.only_load = only_load
        self.my_transform = my_transform
        if data_train_matrix == None:
            self.load_dataset()
            self.build_dataloader()  
            self.convert_to_matrix()
            self.convert_to_rank()
            self.convert_to_matrix()
            self.convert_to_rank()
        else:
            self.train_matrix = data_train_matrix
            self.convert_to_rank()


        # self.plot_vectors()
            
    def load_dataset(self):
        if self.dataset_name == 'CIFAR10':
            self.train_dataset = CIFAR10(root=self.path_to_dataset_folder, train=True, transform=self.my_transform, download=True)
            self.test_dataset =  CIFAR10(root=self.path_to_dataset_folder, train=False, transform=self.my_transform, download=True)
        # 加载其他的数据库，不过我没有MHIST这个数据库，先放在这里
        if self.dataset_name == 'MHIST':
            self.train_dataset = MHIST(root=self.path_to_dataset_folder, train=True, transform=ToTensor(), download=True)
            self.test_dataset =  MHIST(root=self.path_to_dataset_folder, train=False, transform=ToTensor(), download=True)

    # def build_dataloader(self):
    #     self.train_loader = torch.utils.data.DataLoader(self.train_dataset, shuffle=False)
    #     self.test_loader = torch.utils.data.DataLoader(self.test_dataset, shuffle=False)
    #     self.train_data_size = len(self.train_loader)
    #     self.test_data_size = len(self.test_dataset)

    def build_dataloader(self, val_split=0.1):
        # 以 val_split 比例划分验证集和训练集
        val_size = int(val_split * len(self.train_dataset))
        train_size = len(self.train_dataset) - val_size

        # 将 train_dataset 分割成训练集和验证集
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.train_dataset, [train_size, val_size])

        # 创建 DataLoader
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, shuffle=False)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, shuffle=False)

        self.train_data_size = len(self.train_loader)
        self.val_data_size = len(self.val_loader)
        self.test_data_size = len(self.test_loader)
        return self.train_loader, self.val_loader, self.test_loader


    # def convert_to_matrix(self):
    #     train_vectors = []
    #     for images, labels in self.train_loader:
    #         images = images.view(images.size(0), -1)
    #         train_vectors.append(images[0])
    #     self.train_matrix = np.stack(train_vectors)  # 转换为 NumPy 数组
    #     self.train_matrix = torch.tensor(self.train_matrix, dtype=torch.float32).to(device)  # 转移到 GPU


    # def convert_to_rank(self):
    #     self.train_effective_rank = Effective_Ranks(self.train_matrix)
    #     self.train_rk = self.train_effective_rank.get_rk()
    #     self.train_Rk = self.train_effective_rank.get_Rk()
    def convert_to_matrix(self):
        train_vectors = []
        for images, labels in self.train_loader:
            images = images.view(images.size(0), -1)
            train_vectors.append(images[0])
        self.train_matrix = np.stack(train_vectors)
        self.train_matrix = torch.tensor(self.train_matrix, dtype=torch.float32).to(self.device)  # 移动到指定设备

    def convert_to_rank(self):
        self.train_effective_rank = Effective_Ranks(self.train_matrix.to('cpu').numpy())  # 注意要先将 Tensor 移回 CPU
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