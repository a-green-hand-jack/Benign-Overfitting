# 这里是之前发现MLP和LeNet的input计算得到的death len的趋势竟然不一样！！这一点实在是十分的奇怪，所以这里为了验证这个现象，我需要单独的计算一下data本身的betti number的情况

# 加载自己的包
# from TDA.data_tda import Image2TDA, CompareTDA
from nets.net_out_tda import ImageNetTDA, CompareNetTDA
from nets.simple_net import MLP, LeNet, MyLargeCNN
from nets.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

# 加载其他库
import numpy as np
import torchvision.transforms as transforms
import concurrent.futures
import torch
import random


# 首先考虑scale作为增强

def scale_data_tda(scale_path = "./Result/DataTDA", aug_name="scale", model=MLP(), model_name="MLP", betti_dim=1, care_layer=-2):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{model_name}/{aug_name}"

    
    scale_list = np.arange(0.1, 1.02, 0.02)
    for scale in scale_list:
        
        train_transform=transforms.Compose([
                                    transforms.RandomResizedCrop(size=(32,32), scale=(scale, scale)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                                ])
        # train_loader, test_loader = get_cifar10_dataloader(batch_size=64, custom_transform=train_transform)

        print(f"裁切是{scale}.")

        save_floor = f"{scale_path}/{scale*10}/"

        temp_img = ImageNetTDA(costume_transform=train_transform, repetitions=10, save_file_path = save_floor, model=model, betti_dim=betti_dim, care_layer=care_layer)

def angle_data_tda(scale_path = "./Result/DataTDA", aug_name="angle", model=MLP(), model_name="MLP", betti_dim=1, care_layer=-2):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{model_name}/{aug_name}"

    
    min_angle_list = range(0,180,1)
    # for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
    for max_angle in min_angle_list:
        min_angle = -max_angle

        train_transform=transforms.Compose([
                            transforms.RandomRotation(degrees=(min_angle, max_angle)),
                            transforms.ToTensor(),
                            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                            ])
        # train_loader, test_loader = get_cifar10_dataloader(batch_size=64, custom_transform=train_transform)

        print(f"现在的最小角度是{min_angle}，最大角度是{max_angle}.")

        save_floor = f"{scale_path}/{max_angle}/"

        temp_img = ImageNetTDA(costume_transform=train_transform, repetitions=10, save_file_path = save_floor, model=model, betti_dim=betti_dim, care_layer=care_layer)

# 新的处理单个模型的函数
def process_model(model, model_name, scale_path, aug_name):
    if aug_name == "scale":
        scale_data_tda(scale_path=scale_path, model=model, model_name=model_name, aug_name="scale")
    elif aug_name == "angle":
        # print("-"*20)
        angle_data_tda(scale_path=scale_path, model=model, model_name=model_name, aug_name="angle")
        # print("*"*20)

# 修改后的 all_resnet 函数
def all_resnet(aug_name):
    print(f"现在是{aug_name}下的考察")
    # model_list = [ResNet18(), ResNet34(), ResNet50(), ResNet101(), ResNet152()]
    # model_names = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    model_list = [ResNet18(), ResNet34()]
    model_names = ["ResNet18", "ResNet34"]
    # model_list = [MLP()]
    # model_names = ["MLP"]
    scale_path = "./Result/NetTDA_2_1th"

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

    # 设置随机数种子
    seed = 0
    torch.manual_seed(seed)  # 设置torch的随机数种子
    random.seed(seed)  # 设置python的随机数种子
    np.random.seed(seed)  # 设置numpy的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置cuda的随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有cuda设备的随机数种子

    scale_path = "./Result/20231227_new_LeNet"
    model = LeNet()
    model_name = "LeNet_new"
    betti_dim = 1
    care_layer = -2

    angle_data_tda(scale_path=scale_path, model=model, model_name=model_name, aug_name="angle", betti_dim=betti_dim, care_layer=-2)
    # scale_data_tda(scale_path=scale_path, model=model, model_name=model_name, aug_name="scale", betti_dim=betti_dim, care_layer=care_layer)
