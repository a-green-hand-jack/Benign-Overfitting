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

# 加载支持函数和类
from support_code.load_dataset import get_data_loader
from support_code.LeNet import LeNet
from support_code.get_scale import get_scale
from support_code.MLP import MLP

# 设置设备和种子
device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# 测试新的方法


if __name__ == '__main__':

  i = 1
  min_angle_list = range(0,360,10)
  # for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
  for min_angle in min_angle_list:
    max_angle = min_angle + 10

    # print("------------------开始新的循环-----------\nmin_angle={},max_angle={}".format(min_angle, max_angle))
    train_loader, test_loader = get_data_loader(min_angle=0, max_angle=max_angle)
    get_scale(
      train_loader=train_loader, 
      test_loader=test_loader,
      path = "MLP-0-360", 
      num_epochs=80,
      net=MLP(),
      min_angle=0,
      max_angle=max_angle,
      batch_size_train=64,
      batch_size_test=64,
      number=i)
    i += 1
    