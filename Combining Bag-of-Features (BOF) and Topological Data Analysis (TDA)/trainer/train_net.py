# 这里需要定义通用的训练函数，最后返回的就是val上的最大准确率
import torch
import os
from torch.optim.lr_scheduler import CosineAnnealingLR  # 实现cos函数式的变化
from typing import Union, List, Tuple, Any
from torch import optim, nn
from torch.utils.data import DataLoader

def get_best_test_acc(
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    train_loader: DataLoader = None,
    test_loader: DataLoader = None,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    initial_lr: float = 0.01,
    num_epochs: int = 5,
    net: Any = None,
    T_max: Union[None, int] = None,
    batch_size_train: int = 64,
    batch_size_test: int = 64,
    patience = 50
) -> float:
    """
    训练并返回最佳测试准确率。

    Args:
        device (torch.device): 设备 (default: 根据是否可用CUDA自动选择)
        train_loader (DataLoader): 训练数据集的DataLoader
        test_loader (DataLoader): 测试数据集的DataLoader
        momentum (float): SGD动量 (default: 0.9)
        weight_decay (float): 权重衰减 (default: 0.0005)
        initial_lr (float): 初始学习率 (default: 0.01)
        num_epochs (int): 训练轮数 (default: 5)
        net (Any): 网络模型
        T_max (Union[None, int]): 学习率调整的周期 (default: None)
        batch_size_train (int): 训练批次大小 (default: 64)
        batch_size_test (int): 测试批次大小 (default: 64)

    Returns:
        float: 最佳测试准确率
    """
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0
    no_improvement_count = 0

    for epoch in range(num_epochs):
        net.train()
        run_loss = 0
        correct_num = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            out = net(data)[-1]  # 获取最后一层输出
            _, pred = torch.max(out, dim=1)
            optimizer.zero_grad()
            loss = criterion(out, target)
            loss.backward()
            run_loss += loss
            optimizer.step()
            correct_num += torch.sum(pred == target)

        net.eval()
        test_loss = 0
        test_correct_num = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = net(data)[-1]  # 获取最后一层输出
            _, pred = torch.max(out, dim=1)
            test_loss += criterion(out, target)
            test_correct_num += torch.sum(pred == target)

        test_loss = test_loss.item() / len(test_loader)
        test_acc = test_correct_num.item() / (len(test_loader) * batch_size_test)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            no_improvement_count = 0  # 重置计数器
        else:
            no_improvement_count += 1

        # 判断连续未提升次数是否达到设定的耐心值，如果达到则提前停止训练
        if no_improvement_count >= patience:
            print(f"No improvement for {patience} epochs. Stopping early.")
            break

    return best_test_acc

