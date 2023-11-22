from typing import Dict, Union, List, Any, Tuple
import torch
import numpy as np
import os
import pickle
import pandas as pd
import torch.nn.functional as F

from dataset.get_dataloader import get_dataloader,loader2vec, vec_dis
from dataset.data2betti import distance_betti, distance_betti_ripser, plt_betti_number,plot_betti_number_bars
from ripser import Rips, ripser
from net.custome_net import MLP,LeNet,ModelInitializer


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



def get_layer_output_betti(model: Any = None,
                seed: Union[None, int] = None,
                save_root: str = "./distance/Net-test/",
                chose: str = "cifar10_debug",
                debug_size: int = 100,
                device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                name: str = "Net",
                transform: Any = None,
                alpha: float = 0.0,
                gpu_flag: bool = True,
                num_repeats: int = 10) -> Dict[str, Any]:
    ''' 
    Function: get_layer_output_betti

    Description: This function calculates and visualizes the Betti numbers based on the given network, random seed, saving path, dataset type, number of images, Betti number name, and augmentation type.

    Parameters: - model (optional): Neural network model. - seed (optional): Random seed. - save_root (optional): Path to save images and distances. - chose (optional): Selection of dataset type. - debug_size (optional): Number of selected images. - device (optional): Device selection for execution. - name (optional): Name of Betti number. - transform (optional): Selection of data augmentation. - alpha (float): MixUp parameter controlling the degree of mixing. Default is 0.0, indicating no MixUp.

    Returns: A dictionary containing Betti numbers for L1 and L2 norms.

    Example Usage: betti_4_net(model, seed, save_root, chose, debug_size, device, name, transform) 
    '''

    model_initializer = ModelInitializer(model, seed)
    train_loader, test_loader = get_dataloader(chose=chose, debug_size=debug_size, transform=transform)


    model.eval()
    model.to(device)
    # 遍历10遍或者若干遍数据集
    
    all_layers_output = []

    for repeat in range(num_repeats):
        # 遍历数据集并存储每一层的输出
        with torch.no_grad():
            # 初始化一个列表，每个元素都是一个空的张量，用于存储每一层的输出
            layer_outputs = [torch.tensor([]) for _ in range(1 + len(list(model.children())))]
            for batch_idx, (data, target) in enumerate(train_loader):
                # Check whether to perform operations on GPU
                if alpha > 0.0 and alpha <= 0.5:
                    mixed_data = mixup_data(data, alpha)
                else:
                    mixed_data = data

                mixed_data, target = mixed_data.to(device), target.to(device)

                # 获取模型输出
                outputs = model(mixed_data)

                # 将每一层的输出追加到对应的张量中
                for i, layer_output in enumerate(outputs):
                    # 将当前批次的输出连接到之前存储的张量中
                    layer_outputs[i] = torch.cat((layer_outputs[i], layer_output.cpu()), dim=0)

        # 现在 layer_outputs 中每个张量都包含了对应层所有批次输出的连接
        all_layers_output.append(layer_outputs)

    # print(all_layers_output)
    # 两层的list转化为df数据以便于操作
    all_layers_output_df = pd.DataFrame(all_layers_output)
    # print(all_layers_output_df.head())
    # 用于存储每列张量的平均值
    averaged_tensors_list = []

    # 遍历每一列，对每列的张量进行平均值计算
    for column in all_layers_output_df.columns:
        # 将 DataFrame 中的张量转换为 PyTorch 张量
        tensors_in_column = all_layers_output_df[column].values.tolist()
        # tensors_in_column = [torch.tensor(tensor) for tensor in tensors_in_column]

        # 计算每列张量的平均值
        averaged_tensor = torch.stack(tensors_in_column).mean(dim=0)
        averaged_tensors_list.append(averaged_tensor)

    all_l2_distances = []
    all_l1_distances = []
    # print(len(layer_outputs))
    for layer_number, layer_output in enumerate(averaged_tensors_list):
        print(layer_number,"\n",len(layer_output),type(layer_output))
        print(layer_output[0].shape)
        # concatenated_outputs = torch.cat(layer_output, dim=0)
        # print(concatenated_outputs.shape)
        concatenated_outputs = layer_output.view(layer_output.shape[0], -1)
        # print(concatenated_outputs.shape)

        l2_distances = vec_dis(data_matrix=concatenated_outputs, distance="l2", root=save_root, gpu_flag=gpu_flag)
        # print(l2_distances)
        l1_distances = vec_dis(data_matrix=concatenated_outputs, distance="l1", root=save_root, gpu_flag=gpu_flag)
        # print(l1_distances.shape,"\n", l2_distances.shape)

        all_l2_distances.append(l2_distances)
        all_l1_distances.append(l1_distances)

        # 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
        if 'rpp_py' in globals():
            d1 = rpp_py("--format distance --dim 1", l1_distances)
            d2 = rpp_py("--format distance --dim 1", l2_distances)
        else:

            d1 = ripser(l1_distances, maxdim=1, distance_matrix=True)
            d1 = d1["dgms"]
            d2 = ripser(l2_distances, maxdim=1, distance_matrix=True)
            d2 = d2["dgms"]

        save_path =  f"{save_root}/{name}/{layer_number}/"
        plt_betti_number(d1, plt_title=f"{layer_number}L1", root=save_path)
        plot_betti_number_bars(d1, plt_title=f"{layer_number}L1", root=save_path)
        plt_betti_number(d2, plt_title=f"{layer_number}L2", root=save_path)
        plot_betti_number_bars(d2, plt_title=f"{layer_number}L2", root=save_path)
        # print(save_path)

        d1_key = f"{name}-{layer_number}-birth-death-l1_distance"
        d2_key = f"{name}-{layer_number}-birth-death-l2_distance"

        if not os.path.exists(save_path):
            # 如果文件夹不存在，则创建它
            os.makedirs(save_path)

        root = f"{save_path}betti_number.pkl" # 注意带上后缀名
        # 保存字典到文件
        dict_my = {"BD-L1": d1, "BD-L2": d2}

        save_dict(dict_my, root)
    

    return {f"{name}-birth-death-l1_distance": all_l1_distances, f"{name}-birth-death-l2_distance": all_l2_distances}
