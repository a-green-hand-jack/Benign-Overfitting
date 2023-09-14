import torch
from torchvision.transforms import ToTensor
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision
from torchvision.datasets import CIFAR10
import os

# 这里把得到loaddata写成一个函数
def get_data_loader(scale=0.1,
          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124],
          CIFAR_STD = [0.2023, 0.1994, 0.2010],
          valid_scale = 0.1,
          batch_size = 512,
          sub_folder = "dataset_folder",
          root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
          train_transform=None):
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
  # 定义数据预处理的转换，只在训练集上应用
  if train_transform == None:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(32, 32), scale=(scale, scale)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

  # 加载训练集和测试集
  folder_path = os.path.join(root, sub_folder)
  os.makedirs(folder_path, exist_ok=True)
  # 在该子文件夹中保存数据集
  train_dataset = datasets.CIFAR10(root=folder_path, train=True, download=True, transform=train_transform)
  test_dataset = datasets.CIFAR10(root=folder_path, train=False, download=True, transform=transforms.ToTensor())  # 只应用ToTensor

  # 计算分割数据的长度
  num_test = len(test_dataset)
  valid_size = int(valid_scale * num_test)  # 取x%作为验证集

  # 使用random_split来拆分测试集
  valid_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [valid_size, num_test - valid_size])

  # 创建训练集、验证集和测试集的数据加载器
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  return train_loader, valid_loader, test_loader