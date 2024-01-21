import numpy as np
import torch

import random

from ISIC2018.isic_BOF import ISICBOF, CompareBOF
from torchvision import transforms



def process_with_crop(dataset="MONU"):

    if dataset == "BUSI":
        crop_list = range(8, 257, 8)
    elif dataset == "ISIC2018" or "MONU":
        crop_list = range(416, 513, 8)
    for crop_size in crop_list:
        print("-" * 10, crop_size, "-" * 10, "\n")
        
        costume_transform = [transforms.RandomCrop(size=crop_size)]
        # costume_transform = None
        save_pkl_path = f"./Result/new_aug_20240120/{dataset}/{crop_size}/bof.pkl"
        temp_image_processor = ISICBOF(costume_transform=costume_transform, save_file_path=save_pkl_path, repetitions=10, imgpath=f"../Dataset/{dataset}/train_folder")
        # 跳出循环
        # break

def process_with_angle(dataset="MONU"):

    min_angle_list = range(100,180,1)
    # for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
    for max_angle in min_angle_list:
        min_angle = -max_angle
        print(f'最大的角度是{max_angle}')

        costume_transform = [transforms.RandomRotation(degrees=(min_angle, max_angle))]

        # costume_transform = None
        save_pkl_path = f"./Result/new_aug_20240120_angle/{dataset}/{max_angle}/bof.pkl"
        temp_image_processor = ISICBOF(costume_transform=costume_transform, save_file_path=save_pkl_path, repetitions=10, imgpath=f"../Dataset/{dataset}/train_folder")
        # 跳出循环
        # break


       
if __name__ == '__main__':
        # 设置随机数种子
    seed = 0
    torch.manual_seed(seed)  # 设置torch的随机数种子
    random.seed(seed)  # 设置python的随机数种子
    np.random.seed(seed)  # 设置numpy的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置cuda的随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有cuda设备的随机数种子
    # process_with_crop(dataset="BUSI")
    process_with_crop(dataset="ISIC2018")
    # process_with_angle(dataset='BUSI')
    # process_with_angle(dataset='ISIC2018')