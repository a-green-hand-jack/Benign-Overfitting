import torch
from torchvision.transforms import ToTensor
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision
from torchvision.datasets import CIFAR10
import os
import random

def get_data_loader(
    CIFAR_MEAN=[0.49139968, 0.48215827, 0.44653124],
    CIFAR_STD=[0.2023, 0.1994, 0.2010],
    root='./data/',
    batch_size_train=64,
    batch_size_test=64,
    Train_Augment=None
):
    """
    获取用于训练和测试的数据加载器。

    参数:
    - min_angle: 数据增强中的最小旋转角度（默认为None，表示不使用旋转数据增强）
    - max_angle: 数据增强中的最大旋转角度（默认为None，表示不使用旋转数据增强）
    - device: 训练设备，可以是 "cuda" 或 "cpu"（默认根据可用GPU自动选择）
    - CIFAR_MEAN: CIFAR10数据集的均值（默认值为 CIFAR10 的均值）
    - CIFAR_STD: CIFAR10数据集的标准差（默认值为 CIFAR10 的标准差）
    - sub_folder: 数据集存放的子文件夹名称（默认为 "dataset_folder"）
    - root: 数据集根目录（默认为 './data/'）
    - train_transform: 自定义的训练数据变换（默认为None）
    - batch_size_train: 训练数据批量大小（默认为 64）
    - batch_size_test: 测试数据批量大小（默认为 64）
    - RandAugment: 用于数据增强的 RandAugment 对象（默认为None，表示不使用 RandAugment）

    返回:
    - train_loader: 用于训练的数据加载器
    - test_loader: 用于测试的数据加载器
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # 如果提供了 RandAugment，将其插入到数据增强操作的最前面
    if Train_Augment is not None:
        transform_train.transforms.insert(1, Train_Augment)

    # 创建用于训练和测试的数据加载器
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root, train=True, download=True,
            transform=transform_train), batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root, train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])), batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader
