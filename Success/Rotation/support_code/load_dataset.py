import torch
from torchvision.transforms import ToTensor
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision
from torchvision.datasets import CIFAR10
import os
import random

# 这里把得到loaddata写成一个函数
def get_data_loader(
          min_angle = None,
          max_angle = None,
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124],
          CIFAR_STD = [0.2023, 0.1994, 0.2010],
          sub_folder = "dataset_folder",
        #   root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
          root = './data/',
          train_transform=None,
          batch_size_train = 64,
          batch_size_test  = 64):
  '''
  输入: 1. device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),规定了采用的设备,不过似乎没用上
        2. CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124],这是CIFAR10数据集的平均值,因为这个数据集广泛使用,所以是已知的条件
        3. CIFAR_STD = [0.2023, 0.1994, 0.2010]
        4. valid_scale = 0.1,表示要从test_dataset中切割10%作为valid_dataset
        5. batch_size = 5120,表示一次从train_dataset中抽取512张图片作为一个X
        6. sub_folder = "dataset_folder"表示存储下载的文件的文件夹的名字
        7. root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))表示sub_folder所在的文件夹在当前目录的父-父-父文件夹
        8. train_transform = None ,如果没有对train_transform进行额外的要求,那么就采用函数里面默认的
        
  输出: train_loader, valid_loader, test_loader
      只要按顺序接收就好
  '''

  train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.CIFAR10(root, train=True, download=True,
          transform=torchvision.transforms.Compose([
              transforms.RandomRotation(degrees=(min_angle, max_angle)),  # Add random rotation,这里只规定旋转的角度的范围,其余采取默认值,resample=False,采用0来填充旋转之后的区域;expand=False,保持输出的图片大小和输入的图片大小一样,也就是要进行一些剪裁
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
          ])),
      batch_size=batch_size_train, shuffle=True
  )

  
  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root, train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       CIFAR_MEAN, CIFAR_STD)
                               ])),
    batch_size=batch_size_test, shuffle=True)
  # 只有训练集有增强

  # # 创建训练集、验证集和测试集的数据加载器
  # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, test_loader