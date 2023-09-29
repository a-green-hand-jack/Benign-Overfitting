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
# scale_min = 0.1
# scale_max = 1.0
# scale_list = [round(x * 0.1, 1) for x in range(int(scale_min * 10), int(scale_max * 10) + 1)]
# seed = 42
# torch.manual_seed(seed)
# batch_size_train = 64
# batch_size_test  = 64

# # 添加tensorboard
# folder_path = "LeNet"
# writer_new = SummaryWriter(folder_path)

# for scale in scale_list:
#   train_loader, test_loader = get_data_loader(scale=scale)
#   # size = int(32*scale)
#   # print(size)

#   train_tag_loss = "train_loss_{}".format(scale)
#   train_tag_acc = "train_accuracy_{}".format(scale)
#   test_tag_loss = "val_loss_{}".format(scale)
#   test_tag_acc = "val_accuracy_{}".format(scale)
#   print("--------------------开始scale={}的情况---------------------".format(scale), "\n")
#   get_scale(
#             writer=writer_new, 
#             train_loader=train_loader, 
#             test_loader=test_loader,
#             folder_path = folder_path, 
#             test_acc_name= test_tag_acc,
#             test_loss_name=test_tag_loss,
#             train_acc_name=train_tag_acc,
#             train_loss_name=train_tag_loss,
#             num_epochs=500,
#             net=LeNet())


# writer_new.close()

# %%
# 测试新的方法
min_value = 0.1
max_value = 0.1

# 创建一个4x10的矩阵，每一行的内容都是相同的数字序列
matrix = np.array([np.linspace(min_value, max_value, 10)] * 4)

# 定义需要并行执行的函数
def process_scale(bright, contrast, saturation, hue, folder_path="LeNet"):
    # 在这里调用 get_scale() 函数并进行 GPU 运算
    # 注意：需要根据实际情况将 get_scale() 函数的内容替换为你的代码
    train_loader, test_loader = get_data_loader(
          bright_scale=bright,
          contrast_scale=contrast,
          saturation_scale=saturation,
          hue_scale=hue,)
    result = get_scale(
            train_loader=train_loader, 
            test_loader=test_loader,
            folder_path = folder_path, 
            num_epochs=1,
            net=LeNet())
    return result

if __name__ == '__main__':
    mp.set_start_method('spawn')  # 设置多进程启动方式为'spawn'，以便在GPU上运行

    # 使用多进程进行并行计算
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_scale, [(bright, contrast, saturation, hue) for bright in matrix[0] for contrast in matrix[1] for saturation in matrix[2] for hue in matrix[3]])

    # 处理并行计算的结果
    for result in results:
        # 处理每个结果
        pass

