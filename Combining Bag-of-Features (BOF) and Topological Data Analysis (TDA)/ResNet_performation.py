# 这个程序用来实现ResNet系列在angle和scale两种不同的增强下CIFAR10数据集上的准确率的记录

# 首先是加载一些可能用到的包
# 首先加载我自己写的包
from trainer.train_net import get_best_test_acc
from nets.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from nets.simple_net import MLP, LeNet
from TDA.get_dataloader import get_cifar10_dataloader, get_dataloader

# 然后是其他库
import torch
import torchvision.transforms as transforms
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# 首先考虑scale作为增强

def scale_resnet_pre(scale_path = "./Result/ResNetAcc", model=ResNet18(), model_name="ResNet18", aug_name="scale"):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{model_name}/{aug_name}/"

    
    scale_list = np.arange(0.1, 1.02, 0.02)
    for scale in scale_list:
        
        train_transform=transforms.Compose([
                                    transforms.RandomResizedCrop(size=(32,32), scale=(scale, scale)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                                ])
        train_loader, test_loader = get_cifar10_dataloader(batch_size=32, custom_transform=train_transform)
        # train_loader, test_loader = get_dataloader(batch_size=1, transform=train_transform, debug_size=10)
        print(f"裁切是{scale}.")

        save_floor = f"{scale_path}/{scale*10}/"

        best_acc = get_best_test_acc(train_loader=train_loader, test_loader=test_loader, folder_path=save_floor, net=model, net_name=model_name, num_epochs=500)
def scale_resnet_pre_parallel(scale_path = "./Result/ResNetAcc", model=ResNet18(), model_name="ResNet18", aug_name="scale"):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{model_name}/{aug_name}/"

    
    scale_list = np.arange(0.1, 1.02, 0.02)
    def process_scale(scale):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(32, 32), scale=(scale, scale)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
        train_loader, test_loader = get_cifar10_dataloader(batch_size=32, custom_transform=train_transform)
        print(f"裁切是{scale}.")
        save_floor = f"{scale_path}/{scale * 10}/"
        best_acc = get_best_test_acc(train_loader=train_loader, test_loader=test_loader, folder_path=save_floor,
                                     net=model, net_name=model_name, num_epochs=500)

    # 使用 ThreadPoolExecutor 实现并行计算
    with ThreadPoolExecutor() as executor:
        executor.map(process_scale, scale_list)

def angle_resnet_pre(scale_path = "./Result/ResNetAcc", model=ResNet18(), model_name="ResNet18", aug_name="angle"):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{model_name}/{aug_name}/"

    
    min_angle_list = range(0,180,1)
    # for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
    for max_angle in min_angle_list:
        min_angle = -max_angle

        train_transform=transforms.Compose([
                            transforms.RandomRotation(degrees=(min_angle, max_angle)),
                            transforms.ToTensor(),
                            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                            ])
        train_loader, test_loader = get_cifar10_dataloader(batch_size=32, custom_transform=train_transform)
        # train_loader, test_loader = get_dataloader(batch_size=1, transform=train_transform, debug_size=10)

        print(f"现在的最小角度是{min_angle}，最大角度是{max_angle}.")

        save_floor = f"{scale_path}/{max_angle}/"

        best_acc = get_best_test_acc(train_loader=train_loader, test_loader=test_loader, folder_path=save_floor, net=model, net_name=model_name, num_epochs=500)

def angle_resnet_pre_parallel(scale_path = "./Result/ResNetAcc", model=ResNet18(), model_name="ResNet18", aug_name="angle"):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{model_name}/{aug_name}/"

    
    max_angle_list = range(0,180,1)
    # for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
    def process_angle(max_angle):
        min_angle = -max_angle
        train_transform=transforms.Compose([
                            transforms.RandomRotation(degrees=(min_angle, max_angle)),
                            transforms.ToTensor(),
                            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                            ])
        train_loader, test_loader = get_cifar10_dataloader(batch_size=32, custom_transform=train_transform)
        print(f"现在的最小角度是{min_angle}，最大角度是{max_angle}.")
        save_floor = f"{scale_path}/{max_angle}/"
        best_acc = get_best_test_acc(train_loader=train_loader, test_loader=test_loader, folder_path=save_floor, net=model, net_name=model_name, num_epochs=500)
    
     # 使用 ThreadPoolExecutor 实现并行计算
    with ThreadPoolExecutor() as executor:
        executor.map(process_angle, max_angle_list)



# 新的处理单个模型的函数
def process_model(model, model_name, scale_path, aug_name):
    if aug_name == "angle":
        angle_resnet_pre(scale_path=scale_path, model=model, model_name=model_name, aug_name="angle")
    elif aug_name == "scale":
        scale_resnet_pre(scale_path=scale_path, model=model, model_name=model_name, aug_name="scale")

# 修改后的 all_resnet 函数
def all_resnet(aug_name):
    print(f"现在是{aug_name}下的考察")
    model_list = [LeNet()]
    model_names = ["LeNet"]
    scale_path = "./Result/ResNetAcc/new_LeNet"

    with concurrent.futures.ThreadPoolExecutor() as executor:  # 也可以使用 ProcessPoolExecutor 处理 CPU 密集型任务
        futures = []
        for model, model_name in zip(model_list, model_names):
            print("-" * 10, f"现在是{model_name}下的考察", "-" * 10)
            futures.append(
                executor.submit(process_model, model, model_name, scale_path, aug_name)
            )

        # 等待所有任务完成
        concurrent.futures.wait(futures)

if __name__ == '__main__':

    # scale_resnet_pre()
    # angle_resnet_pre()
    # all_resnet(aug_name="angle")
    all_resnet(aug_name="scale")
    # scale_resnet_pre_parallel()
    # scale_resnet_pre()

