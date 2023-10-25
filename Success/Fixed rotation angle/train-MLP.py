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
from support_code.train_model import get_train
from support_code.MLP import MLP

# 设置设备和种子
device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# 测试新的方法


if __name__ == '__main__':

  class RandomBiRotation:
      def __init__(self, degrees):
          self.degrees = degrees

      def __call__(self, img):
          # 随机选择旋转角度（30°或-30°）
          angle = random.choice(self.degrees)
          return transforms.functional.rotate(img, angle)

  
  i = 1
  positive_angle_list = range(22,31,1)
  for positive_angle in positive_angle_list:
    # max_angle = -min_angle

    # print("------------------开始新的循环-----------\nmin_angle={},max_angle={}".format(min_angle, max_angle))
    augment = RandomBiRotation(degrees=[-positive_angle, positive_angle])
    # augment = None
    train_loader, test_loader = get_data_loader(min_angle=-positive_angle, max_angle=positive_angle, RandAugment=augment)
    get_train(
      train_loader=train_loader, 
      test_loader=test_loader,
      path = "MLP-30-30", 
      num_epochs=50,
      net=MLP(),
      min_angle=-positive_angle,
      max_angle=positive_angle,
      batch_size_train=64,
      batch_size_test=64,
      number=i)
    i += 1
    