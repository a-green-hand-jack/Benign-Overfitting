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

# 首先考虑scale作为增强

def scale_resnet_pre(scale_path = "./Result/ResNetAcc", model=ResNet18(), model_name="ResNet18", aug_name="scale", scale_attention_dic={}):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{model_name}/{aug_name}/"

    if scale_attention_dic == {}:
        scale_list = np.arange(0.1, 1.02, 0.02)
    else:
        scale_list = scale_attention_dic[f'{model_name}']
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

def angle_resnet_pre(scale_path = "./Result/ResNetAcc", model=ResNet18(), model_name="ResNet18", aug_name="angle", angle_attention_dic={}):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{model_name}/{aug_name}/"


    if angle_attention_dic == {}:
        min_angle_list = range(0,180,1)
    else:
        min_angle_list = angle_attention_dic[f'{model_name}']
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



# 新的处理单个模型的函数
def process_model(model, model_name, scale_path, aug_name, attention_dic):
    if aug_name == "angle":
        angle_resnet_pre(scale_path=scale_path, model=model, model_name=model_name, aug_name="angle", angle_attention_dic=attention_dic)
    elif aug_name == "scale":
        scale_resnet_pre(scale_path=scale_path, model=model, model_name=model_name, aug_name="scale", scale_attention_dic=attention_dic)

# 修改后的 all_resnet 函数
def all_resnet(aug_name, attention_dic):
    print(f"现在是{aug_name}下的考察")
    model_list = [MLP(), LeNet(), ResNet18(), ResNet34(), ResNet50(), ResNet101(), ResNet152()]
    model_names = ["MLP", "LeNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    scale_path = "./Result/ResNetAcc"

    with concurrent.futures.ThreadPoolExecutor() as executor:  # 也可以使用 ProcessPoolExecutor 处理 CPU 密集型任务
        futures = []
        for model, model_name in zip(model_list, model_names):
            print("-" * 10, f"现在是{model_name}下的考察", "-" * 10)
            futures.append(
                executor.submit(process_model, model, model_name, scale_path, aug_name, attention_dic)
            )

        # 等待所有任务完成
        concurrent.futures.wait(futures)

if __name__ == '__main__':

    # scale_resnet_pre()
    # angle_resnet_pre()
    model_angle_dic = {'MLP':[4,96,138,161,174], 'LeNet':[75,124,163,168], 'ResNet18':[147], 'ResNet34':[147], 'ResNet50':[149], 'ResNet101':[141], 'ResNet152':131}
    
    model_scale_dic = {'MLP':np.arange(0.1, 1.02, 0.02)[::-1][:10], 'LeNet':np.arange(0.1, 1.02, 0.02)[::-1][:10],  'ResNet18':np.arange(0.1, 1.02, 0.02)[::-1][:10], 'ResNet34':[147], 'ResNet50':np.arange(0.1, 1.02, 0.02)[::-1][:10], 'ResNet101':np.arange(0.1, 1.02, 0.02)[::-1][:10], 'ResNet152':np.arange(0.1, 1.02, 0.02)[::-1][:10]}
    
    all_resnet(aug_name="angle", attention_dic=model_angle_dic)
    all_resnet(aug_name="scale", attention_dic=model_scale_dic)

