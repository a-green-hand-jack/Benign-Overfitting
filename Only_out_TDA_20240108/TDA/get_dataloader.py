import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import torch.nn.functional as F
import random

def get_dataloader(chose="cifar10_debug",
                   batch_size=64, 
                           root='./data', 
                           transform=None,
                           debug_size=20,
                           torch_aug=True):
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
        return get_cifar10_dataloader(batch_size=batch_size, root=root, CIFAR_MEAN=CIFAR_MEAN, CIFAR_STD=CIFAR_STD,custom_transform=transform, torch_aug=torch_aug)
    if chose == "cifar10_debug":
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.2023, 0.1994, 0.2010]
        return get_cifar10_debug_dataloader(batch_size=batch_size, root=root, CIFAR_MEAN=CIFAR_MEAN, CIFAR_STD=CIFAR_STD,custom_transform=transform,debug_size=debug_size, torch_aug=torch_aug)


def get_cifar10_dataloader(batch_size=64, 
                           root='./data', 
                           CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124], 
                           CIFAR_STD = [0.2023, 0.1994, 0.2010],
                           custom_transform=None,
                           torch_aug=True):
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
    if torch_aug:
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    else:
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
                                            lambda img: transform(image=np.array(img))['image']
                                        ]), download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False,  transform=transforms.Compose([
                                            lambda img: transform(image=np.array(img))['image']
                                        ]), download=True)


    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar10_debug_dataloader(batch_size=64, 
                                root='./data', 
                                CIFAR_MEAN=[0.49139968, 0.48215827, 0.44653124], 
                                CIFAR_STD=[0.2023, 0.1994, 0.2010],
                                custom_transform=None,
                                debug_size=20,
                                torch_aug=True):
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

    # 加载部分训练集和测试集，随机选择 debug_size 个样本

    # 加载训练集和测试集
    if torch_aug:
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    else:
        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transforms.Compose([
                                            lambda img: transform(image=np.array(img))['image']
                                        ]), download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False,  transform=transforms.Compose([
                                            lambda img: transform(image=np.array(img))['image']
                                        ]), download=True)
        
    indices = random.sample(range(len(train_dataset)), debug_size)
    train_dataset = torch.utils.data.Subset(train_dataset, indices)

    indices = random.sample(range(len(test_dataset)), debug_size)
    test_dataset = torch.utils.data.Subset(test_dataset, indices)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader





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


from typing import Optional
import torch

def loader2vec(train_loader: torch.utils.data.DataLoader, alpha: float = 0.0, gpu_flag: bool = False) -> torch.Tensor:
    """
    将训练数据加载器中的图像批量转换为一维向量并合并成一个大矩阵，根据alpha的值是否使用MixUp。

    参数:
    - train_loader (PyTorch 数据加载器): 用于加载训练数据的数据加载器。
    - alpha (float): MixUp参数，控制混合的程度。默认为0.0，表示不使用MixUp。
    - gpu_flag (bool): 是否使用 GPU 进行计算，如果为 True，则使用 GPU，否则使用 CPU。

    返回:
    - flattened_images (PyTorch Tensor): 一个包含所有训练图像的一维向量矩阵。
    """

    # 检查是否有可用的CUDA设备
    use_cuda = gpu_flag and torch.cuda.is_available()
    
    flattened_images = None

    for images, _ in train_loader:
        # 检查是否应该在GPU上执行
        if use_cuda:
            images = images.to('cuda')

        # 根据alpha的值判断是否应用MixUp
        if 0.0 < alpha <= 0.5:
            mixed_images = mixup_data(images, alpha)
        else:
            mixed_images = images

        # mixed_images是经过MixUp处理的图像，形状为(batch_size, 3, 224, 224)
        
        # 将图像批量转换为一维向量
        batch_flattened = mixed_images.view(mixed_images.size(0), -1)
        
        if flattened_images is None:
            flattened_images = batch_flattened
        else:
            flattened_images = torch.cat((flattened_images, batch_flattened), dim=0)

    print(f"这里是得到图片数值矩阵，是否使用cuda={use_cuda}")
    return flattened_images



from typing import Optional

def vec_dis(data_matrix: torch.Tensor, distance: str, root: str = "./distance", save_flag: Optional[bool] = False, gpu_flag: bool = True):
    """
    计算给定数据矩阵中样本之间的距离矩阵，支持欧氏距离（L2范数）和曼哈顿距离（L1范数）。

    参数:
    - data_matrix (PyTorch Tensor): 包含样本的数据矩阵。
    - distance (字符串): 距离度量，可选值为 "l2" 或 "l1"，分别表示欧氏距离和曼哈顿距离。
    - root (字符串): 距离矩阵保存的根目录。
    - save_flag (布尔值): 是否将距离矩阵保存为文件，如果为 True，则保存。
    - gpu_flag (布尔值): 是否使用 GPU 进行计算，如果为 True，则使用 GPU，否则使用 CPU。

    返回:
    - l_distances (NumPy 数组): 样本之间的距离矩阵。
        同时，这个距离矩阵经过了归一化处理，这样可以在一定程度上保证output得到的betti number 和data直接得到的betti number的具有可比性的。
    """

    # 检查是否有可用的 CUDA 设备
    use_cuda = gpu_flag and torch.cuda.is_available()
    
    # 计算矩阵中每个元素的均值和标准差
    # mean = torch.mean(data_matrix, dim=0)
    # std = torch.std(data_matrix, dim=0)
    # data_matrix = (data_matrix - mean) / std
    data_matrix = data_matrix / torch.norm(data_matrix, p=2, dim=1, keepdim=True)
    # print(f"这里在检查input和output的进行归一化之后的data matrix的第一行，也就是一个样本，他里面的元素应该都在0，1之间:{data_matrix[0]}\n")

    if use_cuda:
        data_matrix = data_matrix.to('cuda')

    if distance == "l2":
        # 计算 L2 范数（欧氏距离）距离矩阵
        l_distances = torch.cdist(data_matrix, data_matrix, p=2)
        # 将对角线元素手动设为零
        mask = torch.eye(l_distances.size(0), dtype=bool).to('cuda')
        l_distances.masked_fill_(mask, 0)
        
        
    elif distance == "l1":
        # 计算 L1 范数（曼哈顿距离）距离矩阵
        l_distances = torch.cdist(data_matrix, data_matrix, p=1)
        # 将对角线元素手动设为零
        mask = torch.eye(l_distances.size(0), dtype=bool).to('cuda')
        l_distances.masked_fill_(mask, 0)

    dis_root = os.path.join(root, f"{distance}_distance.npy")
    if save_flag:
        if not os.path.exists(root):
            # 如果文件夹不存在，则创建它
            os.makedirs(root)
        np.save(dis_root, l_distances.to('cpu').detach().numpy())  # 保存到 CPU 上并转换为 NumPy 数组
    else:
        pass
    
    # print(f"这里是得到距离矩阵，是否使用cuda={use_cuda}")
    # print(f"{distance}_distance.npy被保存在了{root}")
    return l_distances  # 转换为 NumPy 数组并返回


