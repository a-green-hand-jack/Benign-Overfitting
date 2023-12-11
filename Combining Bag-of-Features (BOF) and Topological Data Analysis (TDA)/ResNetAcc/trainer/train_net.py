# 这里需要定义通用的训练函数，最后返回的就是val上的最大准确率
import torch
import os
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau  # 实现cos函数式的变化
from typing import Union, List, Tuple, Any
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_best_test_acc(
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    train_loader: DataLoader = None,
    test_loader: DataLoader = None,
    weight_decay: float = 0.0005,
    initial_lr: float = 0.01,
    num_epochs: int = 5,
    net: Any = None,
    T_max: Union[None, int] = None,
    batch_size_train: int = 64,
    batch_size_test: int = 64,
    patience = 50,
    folder_path:str = 'your_folder_path',
    net_name:str = 'ResNet'
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
    pth_file_path = os.path.join(folder_path, f'{net_name}.pth')
    if os.path.exists(pth_file_path):
        net.load_state_dict(torch.load(pth_file_path))
        print("Loaded trained weights from {}".format(pth_file_path))
    else:
        print("No pre-trained weights found. Starting training from scratch.")
    
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    # 在创建优化器后，初始化 ReduceLROnPlateau 调度器
    # optimizer = optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)

    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=25, verbose=True)

    best_test_acc = 0.0
    no_improvement_count = 0

    # 准备保存训练和验证过程中的指标
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a CSV file in the specified folder to save the train, val loss, and accuracy
    csv_file_path = os.path.join(folder_path, f'{net_name}.csv')
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])
        with tqdm(total=num_epochs) as pbar:
            for epoch in range(num_epochs):
                # 训练部分
                net.train()
                total_loss = 0  # 新增一个变量用于累积损失值
                correct_num = 0

                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    out = net(data)[-1]
                    _, pred = torch.max(out, dim=1)
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()  # 累积每个批次的损失值
                    correct_num += torch.sum(pred == target)

                # 计算平均损失值和准确率
                average_loss = total_loss / len(train_loader)
                train_acc = correct_num.item() / len(train_loader.dataset)

                # 验证部分
                net.eval()
                test_loss = 0
                test_correct_num = 0

                for batch_idx, (data, target) in enumerate(test_loader):
                    data, target = data.to(device), target.to(device)
                    out = net(data)[-1]
                    _, pred = torch.max(out, dim=1)
                    test_loss += criterion(out, target).item()
                    test_correct_num += torch.sum(pred == target)

                # 计算平均测试损失值和准确率
                test_loss /= len(test_loader)
                test_acc = test_correct_num.item() / len(test_loader.dataset)

                # 更新学习率
                # scheduler.step()
                scheduler.step(test_loss)  # 在每个 epoch 结束后更新学习率，根据验证集损失

                # Save train and val loss and accuracy in the CSV file
                with open(csv_file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch+1, average_loss, train_acc, test_loss, test_acc])

                # current_lr = scheduler.last_lr()[0]  # 获取当前学习率
                # current_lr = scheduler.get_last_lr(optimizer)[0]  # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率



                # 更新进度条，显示当前学习率
                pbar.set_description(f'Epoch {epoch+1}, LR: {current_lr:.6f}')  # 将当前学习率添加到描述中
                pbar.set_postfix({'Train Loss': average_loss, 'Train Accuracy': train_acc, 'Val Loss': test_loss, 'Val Accuracy': test_acc})
                pbar.update(1)

                # 比较，确定要不要保留数据和参数
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    no_improvement_count = 0  # 重置计数器
                    # Save the trained parameters in a .pth file in the specified folder
                    # 只用当训练真的提高的时候才会记录参数
                    pth_file_path = os.path.join(folder_path, f'{net_name}.pth')
                    torch.save(net.state_dict(), pth_file_path)
                else:
                    no_improvement_count += 1

                # 判断连续未提升次数是否达到设定的耐心值，如果达到则提前停止训练
                if no_improvement_count >= patience:
                    print(f"No improvement for {patience} epochs. Stopping early.")
                    break

    return best_test_acc

