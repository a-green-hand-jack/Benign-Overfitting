import numpy as np
import torch

import random

from ISIC2018.isic_BOF import ISICBOF, CompareBOF




def process_with_crop(input_height=512, input_width=512):
    for crop_size in range(8, 513, 8):
        # scale_factor = crop_size / max(input_height, input_width)
        print("-" * 10, crop_size, "-" * 10, "\n")
        save_pkl_path = f".\\Result\\MedticlDataset\\isic\\{crop_size}\\bof.pkl"
        temp_image_processor = ISICBOF(costume_transform=None, save_file_path=save_pkl_path, repetitions=5, crop=crop_size)
        # 清空GPU显存
        torch.cuda.empty_cache()

    folder = ".\\Result\\MedticlDataset\\isic"
    test_BOF = CompareBOF(file_path=folder, target_pkl="bof.pkl", aug_name='crop')

    print(test_BOF.comb_BOF)
    test_BOF.draw_BOF(net_name="data", aug_name="crop")


if __name__ == '__main__':
    # 设置随机数种子
    seed = 0
    torch.manual_seed(seed)  # 设置torch的随机数种子
    random.seed(seed)  # 设置python的随机数种子
    np.random.seed(seed)  # 设置numpy的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置cuda的随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有cuda设备的随机数种子
    process_with_crop()