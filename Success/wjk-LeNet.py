# %%
# 加载各种库
import torch
import os 
import torchvision
from torchvision import datasets, transforms
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from torch.optim.lr_scheduler import CosineAnnealingLR  # 实现cos函数式的变化

# 加载支持函数和类
from support_code.load_dataset import get_data_loader
from support_code.LeNet import LeNet
from support_code.get_scale import get_scale
from support_code.MLP import MLP

# 设置设备和种子
device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# 定义初始的scale范围
scale_min = 0.1
scale_max = 1.0
scale_list = [round(x * 0.1, 1) for x in range(int(scale_min * 10), int(scale_max * 10) + 1)]
seed = 42
torch.manual_seed(seed)
batch_size_train = 64
batch_size_test  = 64

# 添加tensorboard
folder_path = "LeNet-0921-1917"
writer_new = SummaryWriter(folder_path)

for scale in scale_list:
  train_loader, test_loader = get_data_loader(scale=scale)
  # size = int(32*scale)
  # print(size)

  train_tag_loss = "train_loss_{}".format(scale)
  train_tag_acc = "train_accuracy_{}".format(scale)
  test_tag_loss = "val_loss_{}".format(scale)
  test_tag_acc = "val_accuracy_{}".format(scale)
  print("--------------------开始scale={}的情况---------------------".format(scale), "\n")
  get_scale(
            writer=writer_new, 
            train_loader=train_loader, 
            test_loader=test_loader,
            folder_path = folder_path, 
            test_acc_name= test_tag_acc,
            test_loss_name=test_tag_loss,
            train_acc_name=train_tag_acc,
            train_loss_name=train_tag_loss,
            num_epochs=500,
            net=LeNet())


writer_new.close()


