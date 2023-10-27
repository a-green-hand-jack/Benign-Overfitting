# %%
# 加载各种库
import torch
import os 
import torchvision
from torchvision import datasets, transforms
import torch.multiprocessing as mp
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR  # 实现cos函数式的变化
from tqdm import tqdm
from PIL import Image

# 加载支持函数和类
from support_code.load_dataset import get_data_loader
from support_code.LeNet import LeNet
from support_code.train_model import get_train
from support_code.MLP import MLP
from torchvision.transforms import functional as F



# 设置设备和种子
device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# 测试新的方法


if __name__ == '__main__':

  i = 10
  my_list = np.arange(0.9, 1.1, 0.1)
  aplpha_list = [round(x, 2) for x in my_list]
  for alpha in aplpha_list:
    # max_angle = -min_angle

    # print("------------------开始新的循环-----------\nmin_angle={},max_angle={}".format(min_angle, max_angle))
    # augment = None
    train_loader, test_loader = get_data_loader()
    get_train(
      train_loader=train_loader, 
      test_loader=test_loader,
      path = "MLP-alpha-uniform-Cos", 
      num_epochs=100,
      net=MLP(),
      file_name= alpha,
      batch_size_train=64,
      batch_size_test=64,
      number=i,
      mixup_transform=alpha)
    # break
    i += 1
    