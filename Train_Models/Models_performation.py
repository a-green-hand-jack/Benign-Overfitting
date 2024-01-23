# 这个程序用来实现ResNet系列在angle和scale两种不同的增强下CIFAR10数据集上的准确率的记录

# 首先是加载一些可能用到的包
# 首先加载我自己写的包
from trainer.train_net import get_best_test_acc
from nets.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from nets.simple_net import MLP, LeNet
from TDA.get_dataloader import get_imagenet_loader

# 然后是其他库
import torch
import torchvision.transforms as transforms
import numpy as np
import concurrent.futures


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

# 首先考虑scale作为增强

def scale_resnet_pre(chose_dataset='tiny-imagenet', 
                    save_path='./BOF_Result/tiny_imagenet/angle', 
                    model=ResNet18(), 
                    model_name="ResNet18", 
                    aug_name="scale"):

    if chose_dataset == 'tiny-imagenet':
        image_size = 64
    elif chose_dataset == 'mini-imagenet':
        image_size = 84
    elif chose_dataset == 'cifar10':
        image_size = 32
    scale_path = f"{save_path}/{model_name}/{aug_name}/"
    # 定义从0.1到0.5的步长为0.02
    scale_list_1 = np.arange(0.1, 0.5 + 0.02, 0.02)

    # 定义从0.5到1的步长为0.05
    scale_list_2 = np.arange(0.5, 1 + 0.05, 0.05)

    # 合并两个列表
    scale_list = np.concatenate((scale_list_1, scale_list_2))
    for scale in scale_list:
        
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(image_size,image_size), scale=(scale, scale)),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])
        train_loader, test_loader = get_imagenet_loader(batch_size=32, train_transform=train_transform, chose_dataset=chose_dataset)
        # train_loader, test_loader = get_dataloader(batch_size=1, transform=train_transform, debug_size=10)
        print(f"裁切是{scale}.")

        save_floor = f"{scale_path}/{scale*10}/"

        best_acc = get_best_test_acc(train_loader=train_loader, test_loader=test_loader, folder_path=save_floor, net=model, net_name=model_name, num_epochs=500)


def angle_resnet_pre(chose_dataset='tiny-imagenet', 
                    save_path='./BOF_Result/tiny_imagenet/', 
                    model=ResNet18(), 
                    model_name="ResNet18", 
                    aug_name="angle"):

    max_angle_list = np.unique(np.round(1.1118 ** np.arange(50)))
    scale_path = f"{save_path}/{model_name}/{aug_name}/"

    for max_angle in max_angle_list:
        min_angle = -max_angle

        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=(min_angle, max_angle)),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])
        train_loader, test_loader = get_imagenet_loader(batch_size=32, train_transform=train_transform, chose_dataset=chose_dataset)
        # train_loader, test_loader = get_dataloader(batch_size=1, transform=train_transform, debug_size=10)

        print(f"现在的最小角度是{min_angle}，最大角度是{max_angle}.")

        save_floor = f"{scale_path}/{max_angle}/"

        best_acc = get_best_test_acc(train_loader=train_loader, test_loader=test_loader, folder_path=save_floor, net=model, net_name=model_name, num_epochs=500)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train different models.')
    parser.add_argument('--model', type=str, choices=['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'MLP', 'LeNet'],help='Choose the model to train.')
    parser.add_argument('--save_path', type=str, default='./Result', help='Path to save the results.')

    args = parser.parse_args()

    model_mapping = {
        'MLP': MLP(im_size=(64, 64)),
        'LeNet': LeNet(input_height=64, input_width=64),
        'ResNet18': ResNet18(),
        'ResNet34': ResNet34(),
        'ResNet50': ResNet50(),
        'ResNet101': ResNet101(),
        'ResNet152': ResNet152()
    }

    if args.model:
        save_path = args.save_path
        model_name = args.model

        if model_name in model_mapping:
            model = model_mapping[model_name]
            # 根据选择的模型进行训练
            scale_resnet_pre(save_path=save_path, model=model, model_name=model_name, aug_name="scale")
            angle_resnet_pre(save_path=save_path, model=model, model_name=model_name, aug_name="angle")
        else:
            print(f"Invalid model choice. Available models: {', '.join(model_mapping.keys())}")
    else:
        print("Please specify the model using --model argument. Available models: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, MLP, LeNet")

