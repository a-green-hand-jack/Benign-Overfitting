# %% [markdown]
# # 关心每一层的变化
# 
# 我发现，MLP和LeNet过滤之后的数据的betti number 表现出的规律性远远超出了data本身的规律性，这是令人难以理解的，有两种方法可以解释：
# 1. data中，`dataloader`加载数据的时候，没有对每一个图片进行单独的增强，而是所有的图片是同样的增强
# 2. `LeNet\MLP`这样的网络确实大大的提取了数据的高维信息，所以反映出了更好的规律性
# 
# 根据目前的信息来看，第一种的概率是很小的，所以我需要在第二种的假设的基础上进行实验，来观察是不是网络在产生作用，
# 为此，我这里将对唯一可以产生数据波动的output的计算部分进行了10次的重复计算并取平均，希望可以以此降低数据的波动情况。
# 同时第一层返回的是没有经过任何处理的图像本身，这样可以很好的观察每一层对数据的影响。

# %%
from dataset.check_betti4net import get_layer_output_betti
from net.custome_net import LeNet, MLP, ResNet18,ResNet34, ResNet50, ResNet101, ResNet152
from dataset.after_betti import get_all_for_betti, after_get_bars

import torchvision.transforms as transforms

# %%
angle_path = "./angle_layer_out/"
image_size = 32
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
min_angle_list = range(0,31,1)
# for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
for max_angle in min_angle_list:
    min_angle = -max_angle
    print(f"\n 现在的最小角度是{min_angle}，最大角度是{max_angle}.\n")

    data_transform={'train':transforms.Compose([
                            transforms.RandomRotation(degrees=(min_angle, max_angle)),
                            transforms.ToTensor(),
                            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                            ])}
    save_floor = f"{angle_path}{max_angle}/"
    model_LeNet = LeNet()
    LeNet_path = get_layer_output_betti(model=model_LeNet, seed=15, save_root=save_floor, name="LeNet")

    model_MLP = MLP()
    MLP_path = get_layer_output_betti(model=model_MLP, seed=15, save_root=save_floor, name="MLP")

    model_ResNet18 = ResNet18()
    ResNet18_path = get_layer_output_betti(model=model_ResNet18, seed=15, save_root=save_floor, name="ResNet18")

    model_ResNet34 = ResNet34()
    ResNet34_path = get_layer_output_betti(model=model_ResNet34, seed=15, save_root=save_floor, name="ResNet34")

    model_ResNet50 = ResNet50()
    ResNet50_path = get_layer_output_betti(model=model_ResNet50, seed=15, save_root=save_floor, name="ResNet50")

    model_ResNet101 = ResNet101()
    ResNet101_path = get_layer_output_betti(model=model_ResNet101, seed=15, save_root=save_floor, name="ResNet101")

    model_ResNet152 = ResNet152()
    ResNet152_path = get_layer_output_betti(model=model_ResNet152, seed=15, save_root=save_floor, name="ResNet152")

    after_get_bars(base_path = MLP_path)

    after_get_bars(base_path = LeNet_path)

    after_get_bars(base_path = ResNet18_path)

    after_get_bars(base_path = ResNet34_path)

    after_get_bars(base_path = ResNet50_path)

    after_get_bars(base_path = ResNet101_path)

    after_get_bars(base_path = ResNet152_path)
# %% 

# # MLP_path = r".\care_layers_output\with_input\MLP"
# after_get_bars(base_path = MLP_path)

# # LeNet_path = r".\care_layers_output\with_input\LeNet"
# after_get_bars(base_path = LeNet_path)

# # ResNet18_path = r".\care_layers_output\with_input\ResNet18"
# after_get_bars(base_path = ResNet18_path)

# # ResNet34_path = r".\care_layers_output\with_input\ResNet34"
# after_get_bars(base_path = ResNet34_path)

# # ResNet50_path = r".\care_layers_output\with_input\ResNet50"
# after_get_bars(base_path = ResNet50_path)

# # ResNet101_path = r".\care_layers_output\with_input\ResNet101"
# after_get_bars(base_path = ResNet101_path)

# # ResNet152_path = r".\care_layers_output\with_input\ResNet152"
# after_get_bars(base_path = ResNet152_path)

# %%
from dataset.transform_AB import compare_after_betti_in_same_augmentation as cabsa
from dataset.transform_AB import get_all_cabs
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示
import os
import pickle

parent_path = angle_path
get_all_cabs(parent_path)

# %%
from dataset.transform_AB import show_all_different
parent_path = angle_path
show_all_different(parent_path)
# %%
