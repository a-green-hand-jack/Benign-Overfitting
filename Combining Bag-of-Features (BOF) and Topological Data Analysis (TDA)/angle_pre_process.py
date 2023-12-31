from trainer.one_bof_tda import ModelWithOneAugmentation
from nets.simple_net import MLP, LeNet
from nets.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
import random
# 释放不需要的内存
torch.cuda.empty_cache()

# %%  这里是一些全局设置
# 在CIFAR10背景下的预定义
image_size = 32
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
# 设置随机数种子
seed = 0
torch.manual_seed(seed)  # 设置torch的随机数种子
random.seed(seed)  # 设置python的随机数种子
np.random.seed(seed)  # 设置numpy的随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)  # 设置cuda的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有cuda设备的随机数种子

angle_path = "./Result/231221/angle_torch"
i = 1
min_png = 6
min_pkl = 1
min_angle_list = range(1,180,1)
# for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
for max_angle in min_angle_list:
    min_angle = -max_angle

    data_transform={'train':transforms.Compose([
                        transforms.RandomRotation(degrees=(min_angle, max_angle)),
                        transforms.ToTensor(),
                        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                        ])}
    data_transform = data_transform["train"]
    # data_transform = A.Compose([
    #         # A.RandomCrop(height=height, width=width, p=1.0),  # 随机裁剪
    #         A.Rotate(limit=max_angle),  # 角度旋转增强，可设置旋转角度的限制
    #         A.Resize(32, 32),  # 调整大小
    #         A.Normalize(),  # 标准化
    #         ToTensorV2()  # 转换为 PyTorch 的 Tensor 格式
    #     ])
    print("*"*10, f"现在的最小角度是{min_angle}，最大角度是{max_angle}", "*"*10)

    save_floor = f"{angle_path}/LeNet/{max_angle}/"

    MLP_no_aug = ModelWithOneAugmentation(model=LeNet(), net_name="LeNet", transform=data_transform, augmentation_name="angle", num_repeats=2, num_epochs=300,save_path=save_floor, train_model=False, torch_aug=True)

    print(MLP_no_aug.betti_features, "\n------------\n")




