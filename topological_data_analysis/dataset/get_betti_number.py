import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from dataset.get_dataloader import get_dataloader,loader2vec, vec_dis
from dataset.data2betti import distance_betti, distance_betti_ripser, plt_betti_number,plot_betti_number_bars
from ripser import Rips, ripser
from net.custome_net import MLP,LeNet,ModelInitializer

def betti_4_net(model=None,seed=None,
                save_root="./distance/Net-test/",
                chose="cifar10_debug",
                debug_size=100,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                name="Net",
                transform=None):
    ''' 
    Function: betti_4_net

    Description: 该函数根据给定的网络、随机数种子、保存路径、数据集类型、图片数量、Betti number的名称和增强方式进行计算和可视化。

    Parameters: - model (可选): 神经网络模型。 - seed (可选): 随机数种子。 - save_root (可选): 图片和距离保存的路径。 - chose (可选): 数据集类型选择。 - debug_size (可选): 选择的图片数量。 - device (可选): 运行设备选择。 - name (可选): Betti number的名称。 - transform (可选): 数据增强方式选择。

    Returns: 一个字典，包含L1和L2范数的Betti number。

    Example Usage: betti_4_net(model, seed, save_root, chose, debug_size, device, name, transform) 
    '''

    model_initializer = ModelInitializer(model, seed)
    train_loader, test_loader = get_dataloader(chose,debug_size,transform=transform)

    model.eval()
    out_list = []
    model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)   # 把数据转移到对应的device上,这里就是cuda
        out = model(data)
        out_list.append(out)


    flattened_images = torch.cat(out_list, dim=0)
    # flattened_images现在包含整个训练集中的图像向量，形状为(N, 3 * 224 * 224)，其中N是训练集的大小
    l2_distances = vec_dis(data_matrix=flattened_images, distance="l2",save_flag=True,root=save_root)
    l1_distances = vec_dis(data_matrix=flattened_images, distance="l1",save_flag=True,root=save_root)

    # 读取 L2 范数距离矩阵
    l2_floor = f"{save_root}l2_distance.npy"
    loaded_l2_distances = np.load(l2_floor)

    # 读取 L1 范数距离矩阵
    l1_floor = f"{save_root}l1_distance.npy"
    loaded_l1_distances = np.load(l1_floor)

    d1= ripser(l1_distances, maxdim=1, distance_matrix=True)
    plt_betti_number(d1["dgms"],plt_title="L1",root=save_root)
    # print(len(d["dgms"][0]))

    plot_betti_number_bars(d1["dgms"],plt_title="L1",root=save_root)

    d2= ripser(l2_distances, maxdim=1, distance_matrix=True)
    plt_betti_number(d2["dgms"],plt_title="L2", root=save_root)
    # print(len(d["dgms"][0]))

    plot_betti_number_bars(d2["dgms"],plt_title="L2",root=save_root)
    
    d1_key = f"{name}-birth-death-l1_distance"
    d2_key = f"{name}-birth-death-l2_distance"
    return {d1_key:d1["dgms"],d2_key:d2["dgms"]} 


def betti_4_data(seed=None,
                save_root="./distance/1000/",
                chose="cifar10_debug",
                debug_size=100,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                name="Net",
                transform=None):
    ''' 
    Function: betti_4_data

    Description: 该函数根据给定的随机数种子、保存路径、数据集类型、图片数量、Betti number的名称和增强方式进行计算和可视化。

    Parameters: - seed (可选): 随机数种子。 - save_root (可选): 图片和距离保存的路径。 - chose (可选): 数据集类型选择。 - debug_size (可选): 选择的图片数量。 - device (可选): 运行设备选择。 - name (可选): Betti number的名称。 - transform (可选): 数据增强方式选择。

    Returns: 一个字典，包含L1和L2范数的Betti number。

    Example Usage: betti_4_data(seed, save_root, chose, debug_size, device, name, transform) 
    '''

    train_loader, test_loader = get_dataloader(chose,debug_size,transform=transform)
    flattened_images = loader2vec(train_loader=train_loader)

    # flattened_images现在包含整个训练集中的图像向量，形状为(N, 3 * 224 * 224)，其中N是训练集的大小
    l2_distances = vec_dis(data_matrix=flattened_images, distance="l2",save_flag=True,root=save_root)
    l1_distances = vec_dis(data_matrix=flattened_images, distance="l1",save_flag=True,root=save_root)

    # 读取 L2 范数距离矩阵
    l2_floor = f"{save_root}l2_distance.npy"
    loaded_l2_distances = np.load(l2_floor)

    # 读取 L1 范数距离矩阵
    l1_floor = f"{save_root}l1_distance.npy"
    loaded_l1_distances = np.load(l1_floor)

    d1= ripser(l1_distances, maxdim=1, distance_matrix=True)
    plt_betti_number(d1["dgms"],plt_title="L1",root=save_root)
    # print(len(d["dgms"][0]))

    plot_betti_number_bars(d1["dgms"],plt_title="L1",root=save_root)

    d2= ripser(l2_distances, maxdim=1, distance_matrix=True)
    plt_betti_number(d2["dgms"],plt_title="L2", root=save_root)
    # print(len(d["dgms"][0]))

    plot_betti_number_bars(d2["dgms"],plt_title="L2",root=save_root)
    
    d1_key = f"{name}-birth-death-l1_distance"
    d2_key = f"{name}-birth-death-l2_distance"
    return {d1_key:d1["dgms"],d2_key:d2["dgms"]} 


if __name__ == '__main__':

    betti_4_net(model=LeNet(), save_root="./distance/LeNet-test/")