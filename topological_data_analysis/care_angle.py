# %% [markdown]
# # 初始化

# %% [markdown]
# ## 便于反复调用

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## 加载函数

# %%
import torch
import torchvision
import torchvision.transforms as transforms
from dataset.get_betti_number import betti_4_data,betti_4_net, check_folder_integrity
from dataset.get_betti_number import check_and_do
from net.custome_net import LeNet, MLP
from dataset.after_betti import after_get_bars
import numpy as np
from tqdm import tqdm  # 如果没有tqdm，可以使用其他进度条库
import os
import pickle
# 释放不需要的内存
torch.cuda.empty_cache()

# 在CIFAR10背景下的预定义
image_size = 32
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

# 为了实现比较而设置的图片加载量和随机数种子
debug_number = 5000
random_seed = 15

# %% [markdown]
# # 考察不同的增强对数据的影响

# %% [markdown]
# ## 考察不同旋转角度

# %% [markdown]
# ### 得到betti_bars

# %%
angle_path = "./angle_distance/angle/data/"
i = 1
min_png = 6
min_pkl = 1
min_angle_list = range(-30,1,1)
# for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
for min_angle in min_angle_list:
    max_angle = -min_angle

    data_transform={'train':transforms.Compose([
                        transforms.RandomRotation(degrees=(min_angle, max_angle)),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                        ])}
    # print(f"\n 现在的最小角度是{min_angle}，最大角度是{max_angle}.\n")

    save_floor = f"{angle_path}{max_angle}/"

    check_and_do(save_floor=save_floor, min_pkl=min_pkl, min_png=min_png, betti_4_data=lambda: betti_4_data(seed=random_seed, save_root=save_floor, debug_size=debug_number,name="data", transform=data_transform["train"]))

# print("\n, 第一次50K数据观察成功，这发生在cpu上！！")

# %% [markdown]
# ### 对betti_bar后处理

# %%
after_get_bars(base_path = angle_path)

# print("\n 第一次50K的数据后处理成功，这发生在cpu上！！")

# %% [markdown]
# # 考察不同增强对LeNet的影响

# %% [markdown]
# ## 考察不同角度

# %% [markdown]
# ### 得到betti_bars

# %%
angle_path = "./angle_distance/angle/LeNet/"
i = 1
min_angle_list = range(-30,1,1)
# for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
for min_angle in min_angle_list:
    max_angle = -min_angle
    data_transform={'train':transforms.Compose([
                        transforms.RandomRotation(degrees=(min_angle, max_angle)),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                        ])}
    # print(f"\n 现在的最小角度是{min_angle}，最大角度是{max_angle}.\n")
    save_floor = f"{angle_path}{max_angle}/"
    check_and_do(save_floor=save_floor, min_png=min_png, min_pkl=min_pkl, betti_4_data=lambda: betti_4_net(model=LeNet(),seed=random_seed, save_root=save_floor, debug_size=debug_number,name="LeNet", transform=data_transform["train"]))

# %% [markdown]
# ### 对betti_bars后处理

# %%
after_get_bars(base_path = angle_path)

# %% [markdown]
# # 考察不同增强对MLP的影响

# %% [markdown]
# ## 考察不同角度

# %% [markdown]
# ### betti number

# %%
angle_path = "./angle_distance/angle/MLP/"
i = 1
min_angle_list = range(-30,1,1)
# for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
for min_angle in min_angle_list:
    max_angle = -min_angle
    data_transform={'train':transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # transforms.Resize(image_size),
                        # transforms.CenterCrop(image_size),
                        transforms.RandomRotation(degrees=(min_angle, max_angle)),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                        ])}
    # print(f"\n 现在的最小角度是{min_angle}，最大角度是{max_angle}.\n")
    save_floor = f"{angle_path}{max_angle}/"
    check_and_do(save_floor=save_floor, min_png=6, min_pkl=1, betti_4_data=lambda: betti_4_net(model=MLP(),seed=random_seed, save_root=save_floor, debug_size=debug_number,name="MLP", transform=data_transform["train"]))
    

# %% [markdown]
# ### after betti number

# %%
after_get_bars(base_path = angle_path)

# %% [markdown]
# # 实现种内对比

# %% [markdown]
# ## 首先是获得种内的数据

# %%
from dataset.transform_AB import compare_after_betti_in_same_augmentation as cabsa
from dataset.transform_AB import get_all_cabs
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示
import os
import pickle

parent_path = ".\\angle_distance"
get_all_cabs(parent_path)

# %% [markdown]
# ## 对种内差异的可视化

# %%
from dataset.transform_AB import show_all_different

show_all_different(parent_path)


