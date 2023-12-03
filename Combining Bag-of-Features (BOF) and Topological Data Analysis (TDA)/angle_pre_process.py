from trainer.one_bof_tda import ModelWithOneAugmentation
from nets.simple_net import MLP, LeNet
from nets.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# 释放不需要的内存
torch.cuda.empty_cache()

# %%  这里是一些全局设置
# 在CIFAR10背景下的预定义
image_size = 32
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

angle_path = "./pre_process_outputs/angle_no_train"
i = 1
min_png = 6
min_pkl = 1
min_angle_list = range(0,45,1)
# for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
for max_angle in min_angle_list:
    min_angle = -max_angle

    data_transform={'train':transforms.Compose([
                        transforms.RandomRotation(degrees=(min_angle, max_angle)),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                        ])}
    data_transform = data_transform["train"]
    print(f"\n 现在的最小角度是{min_angle}，最大角度是{max_angle}.\n")

    save_floor = f"{angle_path}/ResNet50/{max_angle}/"

    MLP_no_aug = ModelWithOneAugmentation(model=ResNet50(), net_name="ResNet50", transform=data_transform, augmentation_name="angle", num_repeats=2, num_epochs=300,save_path=save_floor)

    print(MLP_no_aug.betti_features, "\n------------\n", MLP_no_aug.BOF, "\n----------\n", MLP_no_aug.best_test_acc)




