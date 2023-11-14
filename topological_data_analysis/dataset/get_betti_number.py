import torch
import numpy as np
import os
import pickle
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


def betti_4_net(model=None, seed=None,
                save_root="./distance/Net-test/",
                chose="cifar10_debug",
                debug_size=100,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                name="Net",
                transform=None,
                alpha=0.0,
                gpu_flog=True):
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

    l2_distances = vec_dis(data_matrix=flattened_images, distance="l2", root=save_root,gpu_flag=gpu_flog)
    l1_distances = vec_dis(data_matrix=flattened_images, distance="l1", root=save_root, gpu_flag=gpu_flog)

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
                alpha=0.0,
                gpu_flag = True):
    ''' 
    Function: betti_4_data

    Description: 该函数根据给定的随机数种子、保存路径、数据集类型、图片数量、Betti number的名称和增强方式进行计算和可视化。

    Parameters: - seed (可选): 随机数种子。 - save_root (可选): 图片和距离保存的路径。 - chose (可选): 数据集类型选择。 - debug_size (可选): 选择的图片数量。 - device (可选): 运行设备选择。 - name (可选): Betti number的名称。 - transform (可选): 数据增强方式选择。 - alpha (float): MixUp参数，控制混合的程度。默认为0.0，表示不使用MixUp。

    Returns: 一个字典，包含L1和L2范数的Betti number。

    Example Usage: betti_4_data(seed, save_root, chose, debug_size, device, name, transform) 
    '''

    # train_loader, test_loader = get_dataloader(chose,debug_size,transform=transform)
    train_loader, test_loader = get_dataloader(chose=chose, debug_size=debug_size, transform=transform)
    # print("888888888888888")
    flattened_images = loader2vec(train_loader=train_loader, alpha=alpha, gpu_flag=gpu_flag)

    # flattened_images现在包含整个训练集中的图像向量，形状为(N, 3 * 224 * 224)，其中N是训练集的大小
    l2_distances = vec_dis(data_matrix=flattened_images, distance="l2", root=save_root, gpu_flag=gpu_flag)
    l1_distances = vec_dis(data_matrix=flattened_images, distance="l1", root=save_root, gpu_flag=gpu_flag)

    # 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
    if 'rpp_py' in globals():
        d1 = rpp_py.run("--format distance --dim 1", l1_distances)
        d2 = rpp_py.run("--format distance --dim 1", l2_distances)
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

import os
import pickle
import numpy as np

def is_empty_matrix(matrix):
    return matrix.size == 0

def is_nonempty_dict(obj):
    return isinstance(obj, dict) and bool(obj)

def check_folder_integrity(folder_path: str, min_png: int, min_pkl: int) -> tuple:
    """
    检查指定文件夹的完整性。

    参数:
    - folder_path (str): 要检查的文件夹路径。
    - min_png (int): 要求的最小 .png 文件数量。
    - min_pkl (int): 要求的最小 .pkl 文件数量。

    返回:
    - tuple: 一个包含三个值的元组。
      1. is_integrity (bool): 是否符合完整性要求。
      2. num_png (int): 文件夹中 .png 文件的数量。
      3. num_pkl (int): 文件夹中 .pkl 文件的数量。

    示例:
    >>> folder_path = '.\\distance\\scale\\data\\0.3'
    >>> min_png = 8
    >>> min_pkl = 1
    >>> result = check_folder_integrity(folder_path, min_png, min_pkl)
    >>> print(result)
    (False, 5, 0)
    
    注意事项:
    - 该函数只考虑直接位于指定文件夹下的 .png 和 .pkl 文件，不会遍历子文件夹。
    - 文件夹路径应为字符串类型，最小 .png 和 .pkl 文件数量应为整数。
    """
    # 检查输入路径是否为文件夹
    if not os.path.isdir(folder_path):
        return False, 0, 0

    # 初始化计数器
    num_png = 0
    num_pkl = 0

    # 遍历文件夹
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # 判断是否为 .png 文件
        if file_name.endswith('.png') and os.path.isfile(file_path):
            num_png += 1

        # 判断是否为 .pkl 文件
        elif file_name.endswith('.pkl') and os.path.isfile(file_path):
            num_pkl += 1

            # 检查 .pkl 文件中包含的内容类型
            with open(file_path, 'rb') as pkl_file:
                try:
                    obj = pickle.load(pkl_file)
                    if not is_nonempty_dict(obj) and not is_empty_matrix(obj):
                        num_pkl -= 1  # 不满足条件，减少计数
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
                    num_pkl -= 1  # 不满足条件，减少计数

    # 检查是否符合条件
    is_integrity = num_png >= min_png and num_pkl >= min_pkl

    return is_integrity, num_png, num_pkl



def check_and_do(save_floor: str, min_png: int, min_pkl: int, betti_4_data) -> None:
    # 检查文件夹是否已经存在
    if os.path.exists(save_floor):
        # print(f"已经存在文件夹: {save_floor}")
        check_result, true_png, true_pkl = check_folder_integrity(save_floor, min_png, min_pkl)
        # print(check_result)
    else:
        # 如果文件夹不存在，创建它
        os.makedirs(save_floor)
        print(f"创建文件夹: {save_floor}")
        check_result = False
        true_png = 0
        true_pkl = 0

    if check_result:
        print(f"{save_floor}满足条件，预期最少有{min_png}张图片最少{min_pkl}个betti数据，实际上有{true_png}张图片{true_pkl}份数据，不需要重新计算")        
    else:
        print(f"{save_floor}不满足条件，预期最少有{min_png}张图片最少{min_pkl}个betti数据，但是只有{true_png}张图片{true_pkl}份数据，需要重新计算")
        betti_4_data()

if __name__ == '__main__':

    betti_4_net(model=LeNet(), save_root="./distance/LeNet-test/")