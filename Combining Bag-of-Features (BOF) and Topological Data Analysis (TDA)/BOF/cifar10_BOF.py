# 在ISIC2108上发现了奇怪的现象，我决定在cifar10上验证一下

from BOF.get_rank_from_matrix import Effective_Ranks, Effective_Ranks_GPU

import pickle
from typing import Any, List, Union, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示
import albumentations as A
from PIL import Image
import statistics
import torch
from typing import List, Tuple, Dict, Any, Union
from glob import glob




class ImageProcessor:
    def __init__(self, costume_transform, cifar_path='./data', save_file_path=None, repetitions=1):
        self.train_transform = costume_transform
        self.repetitions = repetitions
        self.cifar10_path = cifar_path

        self.images_matrix = None  # 初始化为 None

        self.BOF_feature_list = self.get_BOF()

        self.BOF_mean_stddev = self.calculate_stats_tensor()

        self.save_stats_to_file(file_path=save_file_path)

    def image_to_vector(self, image):
        image_array = np.array(image)
        augmented_image = self.train_transform(image=image_array)['image']
        image_vector = augmented_image.flatten()  # 转换为向量
        return image_vector

    def images_to_matrix_lists(self):
        transform = self.train_transform

        # Download CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=self.cifar10_path, train=True, download=True, transform=self.train_transform)

        # 创建一个包含随机选取样本的子数据集
        indices = np.random.choice(len(trainset), 5000, replace=False)  # 随机选取 5000 个索引
        sub_trainset = torch.utils.data.Subset(trainset, indices)

        self.images_matrix_lists = []  # Storing multiple image vector matrices

        for _ in range(self.repetitions):
            image_vectors = []  # Storing image vectors for each iteration
            for i in range(len(sub_trainset)):
                image, _ = sub_trainset[i]  # Get image and label (which is not used in this case)
                image_vector = image.view(-1)
                image_vectors.append(image_vector)

            image_matrix = np.stack(image_vectors)
            self.images_matrix_lists.append(image_matrix)  # Append the matrix to the list

    def images_to_matrix_lists_gpu(self):
        transform = self.train_transform
        # Download CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=self.cifar10_path, train=True, download=True,transform=self.train_transform)

        self.images_matrix_lists = []  # Storing multiple image vector matrices

        for _ in range(self.repetitions):
            image_vectors = []  # Storing image vectors for each iteration
            for i in range(5000):
                image, _ = trainset[i]  # Get image and label (which is not used in this case)
                image_gpu = image.cuda() if torch.cuda.is_available() else image  # Move tensor to GPU if available
                image_vector = image_gpu.view(-1)  # Flatten the image tensor
                image_vectors.append(image_vector)
            
            image_matrix = torch.stack(image_vectors).cuda() if torch.cuda.is_available() else torch.stack(image_vectors)  # Convert to PyTorch tensor and move to GPU if available
            self.images_matrix_lists.append(image_matrix)  # Append the matrix to the list

    def get_BOF(self):
        self.images_to_matrix_lists()  # 转换图像到矩阵列表

        results = []
        for image_matrix in self.images_matrix_lists:
            # 使用每个矩阵进行操作，例如 Effective_Ranks
            print(image_matrix.T.shape)
            # get_rank = Effective_Ranks_GPU(image_matrix)
            get_rank = Effective_Ranks(image_matrix)
            r0 = get_rank.r0
            R0 = get_rank.R0
            rk_max_index = get_rank.rk_max_index
            rk_max = get_rank.rk_max_value
            Rk_max = get_rank.Rk_value_max_rk_index

            results.append({"isic": {"r0": r0, "R0": R0, "rk_max_index": rk_max_index, "rk_max": rk_max, "Rk_max": Rk_max}})
        
        return results
    
    def calculate_stats(self):
        data_list = self.BOF_feature_list
        keys = data_list[0]['isic'].keys()  # 获取键名
        results = {}
        for key in keys:
            values = [item['isic'][key] for item in data_list]  # 收集指定键名的所有值
            mean = statistics.mean(values)  # 计算均值
            std_dev = statistics.stdev(values)  # 计算标准差
            results[key] = (mean, std_dev)  # 存储均值和标准差为元组形式

        return results
    
    def calculate_stats_tensor(self) -> Dict[str, Tuple[float, float]]:
        """
        计算 BOF 特征的均值和标准差。
        返回：
        - 包含每个特征计算的均值和标准差的字典。
        """
        data_list = self.BOF_feature_list
        keys = data_list[0]['isic'].keys()

        results = {}
        for key in keys:
            values = [item['isic'][key] for item in data_list]
            values_tensor = torch.tensor(values, dtype=torch.float32)  # 转换为浮点数张量

            mean = torch.mean(values_tensor).item()  # 计算均值并转换为 Python 标量
            std_dev = torch.std(values_tensor).item()  # 计算标准差并转换为 Python 标量

            results[key] = (mean, std_dev)

        return results
    
    def save_stats_to_file(self, file_path):
        # self.BOF_mean_stddev = self.calculate_stats()

        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(file_path, 'wb') as file:
            pickle.dump(self.BOF_mean_stddev, file)



    



class CompareBOF():
    # 这里是为了比较不同的增强强度下的某一种model的表现力

    def __init__(self,
                file_path: str,
                target_pkl: str,
                net_name: str='data',
                aug_name: str='angle'


                ) -> None:
        self.folder_path = file_path
        self.target_pkl = target_pkl

        self.matching_paths = []
        self.find_matching_pkls()

        self.comb_acc = []
        self.comb_acc_matrix = None
        # self.compare_acc()
        self.comb_BOF = []
        self.compare_BOF()

        # self.draw_BOF(net_name=net_name, aug_name=aug_name)
        


        # self.draw_acc(net_name=net_name, aug_name=aug_name/)
        
    def try_load_pkl(self, file_path: str) -> Union[Any, None]:
        """
        尝试加载一个Pickle文件。

        Args:
        - file_path (str): Pickle文件的路径

        Returns:
        - Union[Any, None]: 返回加载的数据，如果出现错误则返回None
        """
        try:
            # 使用绝对路径加载 Pickle 文件
            with open(file_path, 'rb') as file:
                data = pickle.load(file)    # 这里的文件的路径最好使用绝对路径，不然打不开
                # print(f"{file_path} Pickle file loaded successfully.")
                # 这里可以添加你想要做的事情，比如打印数据内容
                # print(data)
                return data
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")
        except Exception as e:
            print("An error occurred:", e)
    
    def find_matching_pkls(self)-> None:
        """
        在给定文件夹路径下搜索与输入的 pkl 文件名相匹配的 pkl 文件，并返回这些文件的路径列表。

        Args:
        - folder_path (str): 要搜索的文件夹路径
        - target_pkl (str): 目标 pkl 文件名

        Returns:
        - list: 匹配的 pkl 文件路径列表
        """
        # 遍历文件夹下的子文件夹
        for root, dirs, files in os.walk(self.folder_path):
            # print('1')
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # 检查子文件夹中的 pkl 文件是否与目标 pkl 文件名匹配
                pkl_files = [f for f in os.listdir(dir_path) if f.endswith('.pkl') and f == self.target_pkl]
                self.matching_paths.extend([os.path.join(dir_path, pkl_file) for pkl_file in pkl_files])
        def get_number_from_path(path):
            # 从路径中提取数字部分
            try:
                number = float(path.split('\\')[-2])
                if number.is_integer():
                    return int(number)
                else:
                    return number
            except ValueError:
                return None  # 或者返回适当的默认值或者标记

        self.matching_paths = sorted(self.matching_paths, key=get_number_from_path)
        # print(self.matching_paths)

    def compare_BOF(self):
        for pkl_path in self.matching_paths:
            temp_get_BOF = self.try_load_pkl(file_path=pkl_path)
            # print(temp_get_acc.feature_cared)
            # value = list(temp_get_acc.values())[0]
            # print('{}'.format(value))
            self.comb_BOF.append(temp_get_BOF)
            # print(temp_get_BOF)
        # self.comb_acc_matrix = np.array(self.comb_acc)


    def draw_BOF(self, net_name, aug_name):
        save_path = os.path.join(self.folder_path, f'BOF_{net_name}_{aug_name}.png')
        data = self.comb_BOF

        # 创建4个子图的大图布局
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))  # 4个子图

        # 遍历每个子图的索引和对应的键
        keys = ['r0', 'R0', 'rk_max_index', 'rk_max', 'Rk_max']  # 四个键
        for idx, subkey in enumerate(keys):
            # 提取每个子图的数据
            # values = [item[subkey][0] for item in data]  # 提取mean值
            # errors = [item[subkey][1] for item in data]  # 提取标准差
            if aug_name == 'scale':
                values = [item[subkey][0] for item in data[::-1]]  # 提取mean值
                errors = [item[subkey][1] for item in data[::-1]]  # 提取标准差
                # # 绘制子图带误差棒
                # axs[idx].errorbar(range(len(values)), values, yerr=errors, linestyle=':', marker='o', markersize=4)
                # axs[idx].set_title(subkey)
                # 绘制子图带误差棒
                axs[idx].errorbar(np.arange(len(values)) / (len(values) - 1), values, yerr=errors, linestyle=':', marker='o', markersize=4)
                axs[idx].set_title(subkey)
            else:
                values = [item[subkey][0] for item in data]  # 提取mean值
                errors = [item[subkey][1] for item in data]  # 提取标准差

                # 绘制子图带误差棒
                axs[idx].errorbar(np.arange(len(values)) / (len(values) - 1), values, yerr=errors, linestyle=':', marker='o', markersize=4)
                axs[idx].set_title(subkey)

        # 调整布局并保存图像
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()
















