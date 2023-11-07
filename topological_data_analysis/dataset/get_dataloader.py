import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

def get_dataloader(chose,batch_size=64, 
                           root='./data', 
                           transform=None):
    if chose == "cifar10":
        CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
        CIFAR_STD = [0.2023, 0.1994, 0.2010]
        return get_cifar10_dataloader(batch_size=batch_size, root=root, CIFAR_MEAN=CIFAR_MEAN, CIFAR_STD=CIFAR_STD,custom_transform=transform)


def get_cifar10_dataloader(batch_size=64, 
                           root='./data', 
                           CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124], 
                           CIFAR_STD = [0.2023, 0.1994, 0.2010],
                           custom_transform=None):
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


def loader2vec(train_loader=None):
    flattened_images = None

    for images, _ in train_loader:
        # images是一个批量的图像，形状为(batch_size, 3, 224, 224)
        
        # 将图像批量转换为一维向量
        batch_flattened = images.view(images.size(0), -1)
        
        if flattened_images is None:
            flattened_images = batch_flattened
        else:
            flattened_images = torch.cat((flattened_images, batch_flattened), dim=0)

    return flattened_images


def vec_dis(data_matrix=None,distance=None,root="./distance", save_flag=None):
    

    if distance == "l2":
        # 计算L2范数（欧氏距离）距离矩阵
        l_distances = torch.cdist(data_matrix, data_matrix, p=2)
        
    elif distance == "l1":
        # 计算L1范数（曼哈顿距离）距离矩阵
        l_distances = torch.cdist(data_matrix, data_matrix, p=1)
    print("得到了{}距离".format(distance))
    
    dis_root = os.path.join(root, f"{distance}_distance.npy")
    if save_flag != None:
        if not os.path.exists(root):
            # 如果文件夹不存在，则创建它
            os.makedirs(root)
        np.save(dis_root, l_distances)
    else:
        pass
    return l_distances