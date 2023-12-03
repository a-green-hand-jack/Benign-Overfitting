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

scale_path = "./pre_process_outputs/scale_no_train"
i = 1
min_png = 6
min_pkl = 1
scale_min = 0.1
scale_max = 1.0
scale_list = [round(x * 0.1, 1) for x in range(int(scale_min * 10), int(scale_max * 10) + 1)]

for scale in scale_list:
    

    transform=transforms.Compose([
                                   transforms.RandomResizedCrop(size=(32,32), scale=(scale, scale)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                               ])
    data_transform = transform
    print(f"\n 裁切是{scale}.\n")

    save_floor = f"{scale_path}/ResNet50/{scale*10}/"

    MLP_no_aug = ModelWithOneAugmentation(model=ResNet50(), net_name="ResNet50", transform=data_transform, augmentation_name="scale", num_repeats=2, num_epochs=300,save_path=save_floor)

    print(MLP_no_aug.betti_features, "\n------------\n", MLP_no_aug.BOF, "\n----------\n", MLP_no_aug.best_test_acc)




