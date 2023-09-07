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
          root='dataset_folder',
          valid_scale = 0.1,
          batch_size = 512,
          sub_folder = "dataset_folder"):
  # 定义数据预处理的转换，只在训练集上应用
  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(size=(32, 32), scale=(scale, scale)),
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
  ])

  # 加载训练集和测试集
  root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#   sub_folder = 'dataset_folder'

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