import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

def get_dataloader(chose,batch_size=64, 
                           root='./data', 
                           transform=None,
                           debug_size=20):
    """
    获取指定数据集的数据加载器。

    参数:
    - chose (字符串): 选择的数据集，可选值为 "cifar10" 或 "cifar10_debug"。
    - batch_size (整数): 批处理大小。
    - root (字符串): 数据集的存储根目录。
    - transform (PyTorch 数据转换): 数据转换操作。
    - debug_size (整数): 仅在 chose 为 "cifar10_debug" 时使用，指定调试时加载的样本数量。

    返回:
    - 如果 chose 为 "cifar10"，返回完整的 CIFAR-10 数据加载器。
    - 如果 chose 为 "cifar10_debug"，返回用于调试的 CIFAR-10 数据加载器，仅加载指定数量的样本。
    """
    if chose == "cifar10":
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.2023, 0.1994, 0.2010]
        return get_cifar10_dataloader(batch_size=batch_size, root=root, CIFAR_MEAN=CIFAR_MEAN, CIFAR_STD=CIFAR_STD,custom_transform=transform)
    if chose == "cifar10_debug":
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.2023, 0.1994, 0.2010]
        return get_cifar10_debug_dataloader(batch_size=batch_size, root=root, CIFAR_MEAN=CIFAR_MEAN, CIFAR_STD=CIFAR_STD,custom_transform=transform,debug_size=debug_size)


def get_cifar10_dataloader(batch_size=64, 
                           root='./data', 
                           CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124], 
                           CIFAR_STD = [0.2023, 0.1994, 0.2010],
                           custom_transform=None):
    """
    获取 CIFAR-10 数据集的训练和测试数据加载器。

    参数:
    - batch_size (整数): 批处理大小。
    - root (字符串): 数据集的存储根目录。
    - CIFAR_MEAN (列表): CIFAR-10 数据集的均值（用于数据标准化）。
    - CIFAR_STD (列表): CIFAR-10 数据集的标准差（用于数据标准化）。
    - custom_transform (PyTorch 数据转换): 自定义数据转换操作，如果为 None，则使用默认转换。

    返回:
    - train_loader (PyTorch 数据加载器): 用于训练集的数据加载器。
    - test_loader (PyTorch 数据加载器): 用于测试集的数据加载器。
    """
    # 定义数据转换
    if custom_transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像数据转换为Tensor
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)  # 标准化数据
        ])
    else:
        transform = custom_transform


    # 加载训练集和测试集
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_cifar10_debug_dataloader(batch_size=64, 
                                root='./data', 
                                CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124], 
                                CIFAR_STD = [0.2023, 0.1994, 0.2010],
                                custom_transform=None,
                                debug_size=20):
    """
    获取 CIFAR-10 数据集的用于调试的训练和测试数据加载器，仅加载指定数量的样本。

    参数:
    - batch_size (整数): 批处理大小。
    - root (字符串): 数据集的存储根目录。
    - CIFAR_MEAN (列表): CIFAR-10 数据集的均值（用于数据标准化）。
    - CIFAR_STD (列表): CIFAR-10 数据集的标准差（用于数据标准化）。
    - custom_transform (PyTorch 数据转换): 自定义数据转换操作，如果为 None，则使用默认转换。
    - debug_size (整数): 仅加载的样本数量，用于调试目的。

    返回:
    - train_loader (PyTorch 数据加载器): 用于训练集的数据加载器，仅包含指定数量的样本。
    - test_loader (PyTorch 数据加载器): 用于测试集的数据加载器，仅包含指定数量的样本。
    """

    # 定义数据转换
    if custom_transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像数据转换为Tensor
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)  # 标准化数据
        ])
    else:
        transform = custom_transform

    # 加载部分训练集和测试集，仅加载 debug_size 个样本
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=False)
    train_dataset.data = train_dataset.data[:debug_size]
    train_dataset.targets = train_dataset.targets[:debug_size]

    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=False)
    test_dataset.data = test_dataset.data[:debug_size]
    test_dataset.targets = test_dataset.targets[:debug_size]

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



import torch

def loader2vec(train_loader=None):
    """
    将训练数据加载器中的图像批量转换为一维向量并合并成一个大矩阵。

    参数:
    - train_loader (PyTorch 数据加载器): 用于加载训练数据的数据加载器。

    返回:
    - flattened_images (PyTorch Tensor): 一个包含所有训练图像的一维向量矩阵。
    """

    # 检查是否有可用的CUDA设备
    use_cuda = torch.cuda.is_available()
    
    flattened_images = None

    for images, _ in train_loader:
        # 检查是否应该在GPU上执行
        if use_cuda:
            images = images.to('cuda')

        # images是一个批量的图像，形状为(batch_size, 3, 224, 224)
        
        # 将图像批量转换为一维向量
        batch_flattened = images.view(images.size(0), -1)
        
        if flattened_images is None:
            flattened_images = batch_flattened
        else:
            flattened_images = torch.cat((flattened_images, batch_flattened), dim=0)

    return flattened_images


import torch

def vec_dis(data_matrix=None, distance=None, root="./distance", save_flag=None):
    """
    计算给定数据矩阵中样本之间的距离矩阵，支持欧氏距离（L2范数）和曼哈顿距离（L1范数）。

    参数:
    - data_matrix (PyTorch Tensor): 包含样本的数据矩阵。
    - distance (字符串): 距离度量，可选值为 "l2" 或 "l1"，分别表示欧氏距离和曼哈顿距离。
    - root (字符串): 距离矩阵保存的根目录。
    - save_flag (布尔值): 是否将距离矩阵保存为文件，如果为 True，则保存。

    返回:
    - l_distances (NumPy 数组): 样本之间的距离矩阵。
    """

    # 检查是否有可用的CUDA设备
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        data_matrix = data_matrix.to('cuda')

    if distance == "l2":
        # 计算L2范数（欧氏距离）距离矩阵
        l_distances = torch.cdist(data_matrix, data_matrix, p=2)
        
    elif distance == "l1":
        # 计算L1范数（曼哈顿距离）距离矩阵
        l_distances = torch.cdist(data_matrix, data_matrix, p=1)

    dis_root = os.path.join(root, f"{distance}_distance.npy")
    if save_flag is not None:
        if not os.path.exists(root):
            # 如果文件夹不存在，则创建它
            os.makedirs(root)
        np.save(dis_root, l_distances.to('cpu').numpy())  # 保存到CPU上并转换为NumPy数组
    else:
        pass

    if use_cuda:
        l_distances = l_distances.to('cpu')  # 将结果移回CPU

    return l_distances.cpu().numpy()  # 转换为NumPy数组并返回

