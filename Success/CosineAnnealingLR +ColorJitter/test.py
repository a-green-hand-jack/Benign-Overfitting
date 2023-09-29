# GPU的并行计算
import numpy as np
import torch
import torch.multiprocessing as mp
import random
import csv
import os
import pandas as pd

def get_scale(
    bright_scale=0.1,
    contrast_scale=0.1,
    saturation_scale=0.1,
    hue_scale=0.1,
    path="test",
    num_epoches = 10  # 你可以将 num_epoches 替换为你需要的值
):
        # 如果目录不存在，创建它
    if not os.path.exists(path):
        os.makedirs(path)
    brightness_factor = random.uniform(bright_scale, bright_scale + 0.01)
    contrast_factor = random.uniform(contrast_scale, contrast_scale + 0.01)
    saturation_factor = random.uniform(saturation_scale, saturation_scale + 0.01)
    hue_factor = random.uniform(hue_scale, hue_scale + 0.01)

    # print("brightness_factor={}".format(brightness_factor))
    # print("contrast_factor={}".format(contrast_factor))
    # print("saturation_factor={}".format(saturation_factor))
    # print("hue_factor={}".format(hue_factor))
    # 定义 num_epoches
    

    # 创建一个字典，初始化值为包含 num_epoches 个零的 NumPy 数组
    scale_dict = {
        "brightness_factor={}".format(brightness_factor): np.zeros(num_epoches),
        "contrast_factor={}".format(contrast_factor): np.zeros(num_epoches),
        "saturation_factor={}".format(saturation_factor): np.zeros(num_epoches),
        "hue_factor={}".format(hue_factor): np.zeros(num_epoches)
    }

    # 使用 Pandas 将 scale_dict 写入 CSV 文件
    df = pd.DataFrame(scale_dict)
    csv_filename = "({:.2f}, {:.2f}, {:.2f}, {:.2f}).csv".format(
        bright_scale, contrast_scale, saturation_scale, hue_scale
    )
    df.to_csv(path + '/' + csv_filename, index=False)




min_value = 0.1
max_value = 0.2

# 创建一个4x10的矩阵，每一行的内容都是相同的数字序列
matrix = np.array([np.linspace(min_value, max_value, 10)] * 4)

# 定义需要并行执行的函数
def process_scale(bright, contrast, saturation, hue):
    # 在这里调用 get_scale() 函数并进行 GPU 运算
    # 注意：需要根据实际情况将 get_scale() 函数的内容替换为你的代码
    result = get_scale(
        bright_scale=bright,
        contrast_scale=contrast,
        saturation_scale=saturation,
        hue_scale=hue
    )
    return result

if __name__ == '__main__':
    mp.set_start_method('spawn')  # 设置多进程启动方式为'spawn'，以便在GPU上运行

    # 使用多进程进行并行计算
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_scale, [(bright, contrast, saturation, hue) for bright in matrix[0] for contrast in matrix[1] for saturation in matrix[2] for hue in matrix[3]])

    # 处理并行计算的结果
    for result in results:
        # 处理每个结果
        pass
