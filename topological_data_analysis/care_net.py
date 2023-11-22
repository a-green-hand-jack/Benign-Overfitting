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
from net.custome_net import LeNet, MLP


model_LeNet = LeNet()
get_layer_output_betti(model=model_LeNet, seed=15, save_root="./care_layers_output/with_input", debug_size=5000, name="LeNet")

model_MLP = MLP()
get_layer_output_betti(model=model_MLP, seed=15, save_root="./care_layers_output/with_input", debug_size=5000, name="MLP")

# %%
from dataset.after_betti import get_all_for_betti, after_get_bars

MLP_path = r".\care_layers_output\with_input\MLP"
after_get_bars(base_path = MLP_path)
# %%
from dataset.after_betti import get_all_for_betti, after_get_bars

LeNet_path = r".\care_layers_output\with_input\LeNet"
after_get_bars(base_path = LeNet_path)

# %%
from dataset.transform_AB import compare_after_betti_in_same_augmentation as cabsa
from dataset.transform_AB import get_all_cabs
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示
import os
import pickle

parent_path = r".\care_layers_output"
get_all_cabs(parent_path)

# %%
from dataset.transform_AB import show_all_different
parent_path = r".\care_layers_output"
show_all_different(parent_path)
# %%
