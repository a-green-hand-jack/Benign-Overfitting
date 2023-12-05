# 这里是为了获得ISIC2018的input image 的BOF的情况

from BOF.get_rank_from_matrix import Effective_Ranks

import os
from PIL import Image
import numpy as np

from PIL import Image
import os
import numpy as np

def images_to_matrix(folder_path):
    image_paths = []  # 存储图片路径
    image_vectors = []  # 存储图片向量

    # 遍历文件夹中的图片文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 只处理特定格式的图片文件
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            
            # 将图像调整为 32x32 大小
            image = image.resize((32, 32))
            
            image_array = np.array(image)
            image_vector = image_array.flatten()  # 转换为向量
            image_paths.append(image_path)
            image_vectors.append(image_vector)

    # 将图片向量堆叠成矩阵
    image_matrix = np.vstack(image_vectors)
    return image_matrix



def get_BOF(images_path):
        
        images_matrix = images_to_matrix(images_path)

        get_rank = Effective_Ranks(images_matrix)

        r0 = get_rank.r0
        R0 = get_rank.R0
        rk_max_index = get_rank.rk_max_index
        rk_max = get_rank.rk_max_value
        Rk_max = get_rank.Rk_value_max_rk_index

        return {f"isic":{"r0":r0, "R0":R0, "rk_max_index":rk_max_index, "rk_max":rk_max, "Rk_max":Rk_max}}