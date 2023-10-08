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
    min_value = 0.1
    max_value = 1.0
    num_columns = 4
    num_repetitions = 5


    # 创建一个4x10的矩阵，每一行的内容都是相同的数字序列
    matrix = np.array([np.linspace(min_value, max_value, int(max_value*num_repetitions))] * num_columns)

    for bright_scale in matrix[0]:
      for contrast_scale in matrix[1]:
        for saturation_scale in matrix[2]:
          for hue_scale in matrix[3]:
              print("------------------开始新的循环-----------\n",bright_scale, contrast_scale, saturation_scale, hue_scale)
              train_loader, test_loader = get_data_loader(
                bright_scale=bright_scale,
                contrast_scale=contrast_scale,
                saturation_scale=saturation_scale,
                hue_scale=hue_scale)
              
              get_scale(
                train_loader=train_loader, 
                test_loader=test_loader,
                path = "LeNet", 
                num_epochs=20,
                net=LeNet(),
                bright_scale=bright_scale,
                contrast_scale=contrast_scale,
                saturation_scale=saturation_scale,
                hue_scale=hue_scale)

