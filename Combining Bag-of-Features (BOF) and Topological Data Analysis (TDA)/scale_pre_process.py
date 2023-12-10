from trainer.one_bof_tda import ModelWithOneAugmentation
from nets.simple_net import MLP, LeNet
from nets.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
# 释放不需要的内存
torch.cuda.empty_cache()

# %%  这里是一些全局设置
# 在CIFAR10背景下的预定义
image_size = 32
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

scale_path = "./test_rescale_1207/scale_torch"
i = 1
min_png = 6
min_pkl = 1
scale_min = 0.1
scale_max = 1.0
# scale_list = [round(x * 0.1, 1) for x in range(int(scale_min * 10), int(scale_max * 10) + 1)]
scale_list = np.arange(0.1, 1.02, 0.02)
input_height = 32
input_width = 32

for scale in scale_list:
    

    train_transform=transforms.Compose([
                                   transforms.RandomResizedCrop(size=(32,32), scale=(scale, scale)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                               ])

    # height = int(scale * input_height)
    # width = int(scale * input_width)
    # train_transform = A.Compose([
    #         A.RandomCrop(height=height, width=width, p=1.0),  # 随机裁剪
    #         A.Resize(32, 32),  # 调整大小
    #         A.Normalize(),  # 标准化
    #         ToTensorV2(),
    #     ])

    print(f"\n 裁切是{scale}.\n")

    save_floor = f"{scale_path}/MLP/{scale*10}/"

    MLP_no_aug = ModelWithOneAugmentation(model=MLP(), net_name="MLP", transform=train_transform, augmentation_name="scale", num_repeats=2, num_epochs=300,save_path=save_floor, train_model=True, torch_aug=True)

    print(MLP_no_aug.betti_features, "\n------------\n")




