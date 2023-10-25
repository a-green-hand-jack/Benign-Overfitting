# %%
# 加载各种库
import torch
import os 
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
from support_code.CNN_cls import CNN_cls
from support_code.get_scale import get_scale

# 设置设备和种子
device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 设置随机种子
seed = 10  # 你可以选择任何整数作为种子值
# 设置PyTorch随机种子
torch.manual_seed(seed)
# 设置CUDA随机种子（如果使用GPU）
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# 设置Numpy随机种子
np.random.seed(seed)
# 设置Python随机种子
random.seed(seed)



# %%
# 定义初始的scale范围
scale_min = 1.0
scale_max = 1.0
scale_list = [round(x * 0.1, 1) for x in range(int(scale_min * 10), int(scale_max * 10) + 1)]

# 添加tensorboard
folder_path = "LeNet-0921-1305"
writer_new = SummaryWriter(folder_path)

for scale in scale_list:
  train_loader, _, test_loader = get_data_loader(scale, valid_scale=0.1)
  train_tag_loss = "train_loss_{}".format(scale)
  train_tag_acc = "train_accuracy_{}".format(scale)
  test_tag_loss = "val_loss_{}".format(scale)
  test_tag_acc = "val_accuracy_{}".format(scale)
  print("--------------------开始scale={}的情况---------------------".format(scale), "\n")
  get_scale(scale=scale, 
            writer=writer_new, 
            train_loader=train_loader, 
            test_loader=test_loader,
            folder_path = folder_path, 
            test_acc_name= test_tag_acc,
            test_loss_name=test_tag_loss,
            train_acc_name=train_tag_acc,
            train_loss_name=train_tag_loss,
            num_epochs=100,
            net=CNN_cls(3))


writer_new.close()


