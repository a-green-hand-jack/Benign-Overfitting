import torch

import numpy as np

from dataset.get_dataloader import get_dataloader,loader2vec, vec_dis
from dataset.data2betti import distance_betti, distance_betti_ripser, plt_betti_number,plot_betti_number_bars
from ripser import Rips, ripser
from net.custome_net import MLP,LeNet,ModelInitializer
import os
import pickle
import torch.nn.functional as F

# 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
try:
    import ripserplusplus as rpp_py
except ImportError:
    from ripser import ripser


def save_dict(dictionary, file_path):
    '''
    将字典保存到文件中。

    参数：
    - dictionary：要保存的字典。
    - file_path：文件路径。
    '''
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)


def mixup_data(x, alpha=1.0):
    """
    MixUp数据。将输入数据进行MixUp处理。

    参数：
    - x (PyTorch Tensor): 输入数据。
    - alpha (float): MixUp参数，控制混合的程度。默认为1.0，表示完全使用MixUp。

    返回：
    - mixed_x (PyTorch Tensor): MixUp处理后的数据。
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x


def betti_4_net(model=None, seed=None,
                save_root="./distance/Net-test/",
                chose="cifar10_debug",
                debug_size=100,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                name="Net",
                transform=None,
                alpha=0.0):
    ''' 
    Function: betti_4_net

    Description: 该函数根据给定的网络、随机数种子、保存路径、数据集类型、图片数量、Betti number的名称和增强方式进行计算和可视化。

    Parameters: - model (可选): 神经网络模型。 - seed (可选): 随机数种子。 - save_root (可选): 图片和距离保存的路径。 - chose (可选): 数据集类型选择。 - debug_size (可选): 选择的图片数量。 - device (可选): 运行设备选择。 - name (可选): Betti number的名称。 - transform (可选): 数据增强方式选择。 - alpha (float): MixUp参数，控制混合的程度。默认为0.0，表示不使用MixUp。

    Returns: 一个字典，包含L1和L2范数的Betti number。

    Example Usage: betti_4_net(model, seed, save_root, chose, debug_size, device, name, transform) 
    '''

    model_initializer = ModelInitializer(model, seed)
    train_loader, test_loader = get_dataloader(chose=chose, debug_size=debug_size, transform=transform)

    model.eval()
    out_list = []
    model.to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        # 检查是否应该在GPU上执行
        if alpha > 0.0 and alpha <= 0.5:
            mixed_data = mixup_data(data, alpha)
        else:
            mixed_data = data

        mixed_data, target = mixed_data.to(device), target.to(device)   # 把数据转移到对应的device上，这里就是cuda
        out = model(mixed_data)
        out_list.append(out)

    flattened_images = torch.cat(out_list, dim=0)
    # flattened_images现在是经过网络处理的最后的output层，没有经过softmax，形状为(N,210)，其中N是训练集的大小

    l2_distances = vec_dis(data_matrix=flattened_images, distance="l2", root=save_root)
    l1_distances = vec_dis(data_matrix=flattened_images, distance="l1", root=save_root)

    # 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
    if 'rpp_py' in globals():
        d1 = rpp_py("--format distance --dim 1", l1_distances)
        d2 = rpp_py("--format distance --dim 1", l2_distances)
    else:

        d1 = ripser(l1_distances, maxdim=1, distance_matrix=True)
        d1 = d1["dgms"]
        d2 = ripser(l2_distances, maxdim=1, distance_matrix=True)
        d2 = d2["dgms"]

    plt_betti_number(d1, plt_title="L1", root=save_root)
    plot_betti_number_bars(d1, plt_title="L1", root=save_root)
    plt_betti_number(d2, plt_title="L2", root=save_root)
    plot_betti_number_bars(d2, plt_title="L2", root=save_root)


    d1_key = f"{name}-birth-death-l1_distance"
    d2_key = f"{name}-birth-death-l2_distance"

    if not os.path.exists(save_root):
        # 如果文件夹不存在，则创建它
        os.makedirs(save_root)

    root = f"{save_root}/betti_number.pkl" # 注意带上后缀名
    # 保存字典到文件
    dict_my = {"BD-L1": d1, "BD-L2": d2}

    save_dict(dict_my, root)
    
    return {d1_key: d1, d2_key: d2}


def betti_4_data(seed=None,
                save_root="./distance/1000/",
                chose="cifar10_debug",
                debug_size=100,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                name="Net",
                transform=None,
                alpha=0.0):
    ''' 
    Function: betti_4_data

    Description: 该函数根据给定的随机数种子、保存路径、数据集类型、图片数量、Betti number的名称和增强方式进行计算和可视化。

    Parameters: - seed (可选): 随机数种子。 - save_root (可选): 图片和距离保存的路径。 - chose (可选): 数据集类型选择。 - debug_size (可选): 选择的图片数量。 - device (可选): 运行设备选择。 - name (可选): Betti number的名称。 - transform (可选): 数据增强方式选择。 - alpha (float): MixUp参数，控制混合的程度。默认为0.0，表示不使用MixUp。

    Returns: 一个字典，包含L1和L2范数的Betti number。

    Example Usage: betti_4_data(seed, save_root, chose, debug_size, device, name, transform) 
    '''

    # train_loader, test_loader = get_dataloader(chose,debug_size,transform=transform)
    train_loader, test_loader = get_dataloader(chose=chose, debug_size=debug_size, transform=transform)
    flattened_images = loader2vec(train_loader=train_loader, alpha=alpha)

    # flattened_images现在包含整个训练集中的图像向量，形状为(N, 3 * 224 * 224)，其中N是训练集的大小
    l2_distances = vec_dis(data_matrix=flattened_images, distance="l2", root=save_root)
    l1_distances = vec_dis(data_matrix=flattened_images, distance="l1", root=save_root)

    # 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
    if 'rpp_py' in globals():
        d1 = rpp_py("--format distance --dim 1", l1_distances)
        d2 = rpp_py("--format distance --dim 1", l2_distances)
    else:

        d1 = ripser(l1_distances, maxdim=1, distance_matrix=True)
        d1 = d1["dgms"]
        d2 = ripser(l2_distances, maxdim=1, distance_matrix=True)
        d2 = d2["dgms"]

    plt_betti_number(d1, plt_title="L1", root=save_root)
    plot_betti_number_bars(d1, plt_title="L1", root=save_root)
    plt_betti_number(d2, plt_title="L2", root=save_root)
    plot_betti_number_bars(d2, plt_title="L2", root=save_root)


    d1_key = f"{name}-birth-death-l1_distance"
    d2_key = f"{name}-birth-death-l2_distance"

    if not os.path.exists(save_root):
        # 如果文件夹不存在，则创建它
        os.makedirs(save_root)

    root = f"{save_root}/betti_number.pkl" # 注意带上后缀名
    # 保存字典到文件
    dict_my = {"BD-L1": d1, "BD-L2": d2}

    save_dict(dict_my, root)
    
    return {d1_key: d1, d2_key: d2}



if __name__ == '__main__':

    betti_4_net(model=LeNet(), save_root="./distance/LeNet-test/")