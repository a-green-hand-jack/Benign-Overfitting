# 加载各种库
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR  # 实现cos函数式的变化
from support_code.LeNet import LeNet
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_train(
    # 数据加载器的设备（默认为 GPU 如果可用，否则为 CPU）
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    train_loader=None,
    test_loader=None,
    momentum=0.9,  # 动量参数
    weight_decay=0.0005,  # 权重衰减参数
    initial_lr=0.01,  # 初始学习率
    num_epochs=50,  # 总训练周期数
    net=LeNet(),  # 神经网络模型
    T_max=None,  # CosineAnnealingLR 调度的 T_max 参数
    batch_size_train=64,  # 训练数据批量大小
    batch_size_test=64,  # 测试数据批量大小
    min_angle=None,  # 数据增强中的最小旋转角度
    max_angle=None,  # 数据增强中的最大旋转角度
    scheduler=None,  # 学习率调度器
    optimizer=None,  # 优化器
    path="scale",  # CSV 文件保存路径
    number=None  # 数字标识
):
    """

    参数:
    - device: 数据加载器的设备，可以是 "cuda" 或 "cpu"（默认根据可用 GPU 自动选择）
    - train_loader: 用于训练的数据加载器
    - test_loader: 用于测试的数据加载器
    - momentum: 优化器的动量参数（默认为 0.9）
    - weight_decay: 权重衰减参数（默认为 0.0005）
    - initial_lr: 初始学习率（默认为 0.01）
    - num_epochs: 总训练周期数（默认为 50）
    - net: 神经网络模型（默认为 LeNet）
    - T_max: CosineAnnealingLR 调度的 T_max 参数
    - batch_size_train: 训练数据批量大小（默认为 64）
    - batch_size_test: 测试数据批量大小（默认为 64）
    - min_angle: 数据增强中的最小旋转角度
    - max_angle: 数据增强中的最大旋转角度
    - scheduler: 学习率调度器
    - optimizer: 优化器
    - path: 保存 CSV 文件的路径（默认为 "scale"）
    - number: 数字标识

    返回:
    - 保存了学习率、损失和准确率数据的 CSV 文件

    Args:
    - device: Device for data loaders, can be "cuda" or "cpu" (defaults to automatic GPU selection if available)
    - train_loader: Data loader for training
    - test_loader: Data loader for testing
    - momentum: Momentum parameter for the optimizer (defaults to 0.9)
    - weight_decay: Weight decay parameter for the optimizer (defaults to 0.0005)
    - initial_lr: Initial learning rate (defaults to 0.01)
    - num_epochs: Total number of training epochs (defaults to 50)
    - net: Neural network model (defaults to LeNet)
    - T_max: T_max parameter for CosineAnnealingLR scheduler
    - batch_size_train: Training data batch size (defaults to 64)
    - batch_size_test: Testing data batch size (defaults to 64)
    - min_angle: Minimum rotation angle for data augmentation
    - max_angle: Maximum rotation angle for data augmentation
    - scheduler: Learning rate scheduler
    - optimizer: Optimizer
    - path: Path to save CSV files (defaults to "scale")
    - number: Numeric identifier

    Returns:
    - CSV file containing learning rate, loss, and accuracy data
    
    """
    # 创建 CSV 文件保存路径
    path = path + "/csv"
    if not os.path.exists(path):
        os.makedirs(path)

    # 将神经网络模型移到指定的设备上（如 GPU）
    net.to(device)

    # 如果没有提供优化器，创建一个 SGD 优化器
    if optimizer is None:
        optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 如果没有提供学习率调度器，创建一个 CosineAnnealingLR 调度器
    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=int(num_epochs * len(train_loader)))

    # 创建一个字典来收集数据
    scale_dict = {
        "learning-rate": np.zeros(num_epochs),
        "train_loss": np.zeros(num_epochs),
        "train_acc": np.zeros(num_epochs),
        "test_loss": np.zeros(num_epochs),
        "test_acc": np.zeros(num_epochs)
    }

    for epoch in tqdm(range(num_epochs), unit="epoch", desc="Training"):
        # 设置神经网络为训练模式
        net.train()
        run_loss = 0
        correct_num = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            out = net(data)
            _, pred = torch.max(out, dim=1)
            optimizer.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            run_loss += loss
            optimizer.step()
            scheduler.step()
            correct_num += torch.sum(pred == target)

        train_loss = run_loss.item() / len(train_loader)
        train_acc = correct_num.item() / (len(train_loader) * batch_size_train)
        scale_dict["train_loss"][epoch] = train_loss
        scale_dict["train_acc"][epoch] = train_acc
        scale_dict["learning-rate"][epoch] = optimizer.param_groups[0]['lr']

        # 设置神经网络为评估模式
        net.eval()
        test_loss = 0
        test_correct_num = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = net(data)
            _, pred = torch.max(out, dim=1)
            test_loss += criterion(out, target)
            test_correct_num += torch.sum(pred == target)

        test_loss = test_loss.item() / len(test_loader)
        test_acc = test_correct_num.item() / (len(test_loader) * batch_size_test)

        # 将数据保存到字典中
        scale_dict["test_loss"][epoch] = test_loss
        scale_dict["test_acc"][epoch] = test_acc

    # 使用 Pandas 将数据保存到 CSV 文件
    df = pd.DataFrame(scale_dict)
    csv_filename = "{}-{:.4f}, {:.4f}.csv".format(number, min_angle, max_angle)
    df.to_csv(path + '/' + csv_filename, index=False)
