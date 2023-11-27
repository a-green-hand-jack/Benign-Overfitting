# 这里是为了将增强和隐藏层输出交联起来，最后产生的是3维的可以交互的图像

# %% 首先需要引入必要的库
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm  # 如果没有tqdm，可以使用其他进度条库
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示
import os
import pickle
# 引入自己定义的辅助脚本
from dataset.get_betti_number import betti_4_data,betti_4_net, check_folder_integrity
from dataset.get_betti_number import check_and_do
from net.custome_net import LeNet, MLP
from dataset.after_betti import after_get_bars
from dataset.check_betti4net import get_layer_output_betti
from net.custome_net import LeNet, MLP, ResNet18,ResNet34, ResNet50, ResNet101, ResNet152
from dataset.after_betti import get_all_for_betti, after_get_bars
from dataset.transform_AB import compare_after_betti_in_same_augmentation as cabsa
from dataset.transform_AB import get_all_cabs, show_all_different
from dataset.get_betti_number import check_and_do


# 释放不需要的内存
torch.cuda.empty_cache()

# %%  这里是一些全局设置
# 在CIFAR10背景下的预定义
image_size = 32
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

# 为了实现比较而设置的图片加载量和随机数种子
debug_number = 1000

# %% 考察裁切的变化
scale_path = "./scale_layer_out"
i = 1
scale_min = 0.1
scale_max = 1.0
min_png = 6
min_pkl = 1

scale_list = [round(x * 0.1, 1) for x in range(int(scale_min * 10), int(scale_max * 10) + 1)]

for scale in scale_list:
    data_transform=transforms.Compose([transforms.RandomResizedCrop(size=(32,32), scale=(scale, scale)),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    save_floor = f"{scale_path}{scale}/"
    print(f"\n 现在的裁切大小是{scale}.\n")

    save_floor = f"{scale_path}{scale}/"

    model_LeNet = LeNet()
    LeNet_path = f"{scale_path}/LeNet/{scale}"
    check_and_do(save_floor=LeNet_path, min_pkl=min_pkl, min_png=min_png, betti_4_data=lambda: get_layer_output_betti(model=model_LeNet, seed=15, save_root=LeNet_path, name="LeNet", debug_size=debug_number, transform=data_transform))
    after_get_bars(base_path = LeNet_path)

    model_MLP = MLP()
    MLP_path = f"{scale_path}/MLP/{scale}"
    check_and_do(save_floor=MLP_path, min_pkl=min_pkl, min_png=min_png, betti_4_data=lambda: get_layer_output_betti(model=model_MLP, seed=15, save_root=MLP_path, name="MLP", debug_size=debug_number, transform=data_transform))
    after_get_bars(base_path = MLP_path)

    model_ResNet18 = ResNet18()
    ResNet18_path = f"{scale_path}/ResNet18/{scale}"
    check_and_do(save_floor=ResNet18_path, min_pkl=min_pkl, min_png=min_png, betti_4_data=lambda: get_layer_output_betti(model=model_ResNet18, seed=15, save_root=ResNet18_path, name="ResNet18", debug_size=debug_number, transform=data_transform))
    after_get_bars(base_path=ResNet18_path)

    model_ResNet34 = ResNet34()
    ResNet34_path = f"{scale_path}/ResNet34/{scale}"
    check_and_do(save_floor=ResNet34_path, min_pkl=min_pkl, min_png=min_png, betti_4_data=lambda: get_layer_output_betti(model=model_ResNet34, seed=15, save_root=ResNet34_path, name="ResNet34", debug_size=debug_number, transform=data_transform))
    after_get_bars(base_path=ResNet34_path)

    model_ResNet50 = ResNet50()
    ResNet50_path = f"{scale_path}/ResNet50/{scale}"
    check_and_do(save_floor=ResNet50_path, min_pkl=min_pkl, min_png=min_png, betti_4_data=lambda: get_layer_output_betti(model=model_ResNet50, seed=15, save_root=ResNet50_path, name="ResNet50", debug_size=debug_number, transform=data_transform))
    after_get_bars(base_path=ResNet50_path)

    model_ResNet101 = ResNet101()
    ResNet101_path = f"{scale_path}/ResNet101/{scale}"
    check_and_do(save_floor=ResNet101_path, min_pkl=min_pkl, min_png=min_png, betti_4_data=lambda: get_layer_output_betti(model=model_ResNet101, seed=15, save_root=ResNet101_path, name="ResNet101", debug_size=debug_number, transform=data_transform))
    after_get_bars(base_path=ResNet101_path)

    model_ResNet152 = ResNet152()
    ResNet152_path = f"{scale_path}/ResNet152/{scale}"
    check_and_do(save_floor=ResNet152_path, min_pkl=min_pkl, min_png=min_png, betti_4_data=lambda: get_layer_output_betti(model=model_ResNet152, seed=15, save_root=ResNet152_path, name="ResNet152", debug_size=debug_number, transform=data_transform))
    after_get_bars(base_path=ResNet152_path)

# %% 
parent_path = scale_path
get_all_cabs(parent_path)


show_all_different(parent_path)