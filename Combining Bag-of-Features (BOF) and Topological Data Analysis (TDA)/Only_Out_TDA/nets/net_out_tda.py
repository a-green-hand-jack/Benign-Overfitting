# 首先加载自己定义的包
from TDA.get_dataloader import get_cifar10_dataloader, get_dataloader,loader2vec, vec_dis
from TDA.after_betti import calculate_edge_length, get_min_max_columns, count_epsilon_bar_number,get_max_death

# 然后加载其他库
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.utils.data import Subset
import pickle
from typing import Any, List, Union, Dict
import os
import matplotlib.pyplot as plt
import torchvision
from nets.simple_net import MLP, LeNet
from nets.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示
import statistics
# 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
try:
    import ripserplusplus as rpp_py
except ImportError:
    from ripser import ripser
# 对模型参数进行高斯分布的随机初始化
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.linear import Linear
# 一些辅助函数和类

def init_weights(m: nn.Module) -> None:
    """
    初始化神经网络模型的权重和偏置（如果存在）。

    Args:
    - m (nn.Module): 需要初始化权重和偏置的神经网络模型

    Returns:
    - None
    """
    if isinstance(m, (_ConvNd, Linear)):  # 检查是否是卷积层或线性层
        init.normal_(m.weight.data, mean=0, std=0.01)  # 初始化权重为均值为0，标准差为0.01的正态分布
        if isinstance(m, _ConvNd) and m.bias is not None:
            init.constant_(m.bias.data, 0)  # 如果是卷积层且有偏置项，初始化偏置为常数0
        elif isinstance(m, Linear) and hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0)  # 如果是线性层且有偏置项，初始化偏置为常数0

class ImageNetTDA:
    def __init__(self, costume_transform, cifar_path='./data', save_file_path=None, repetitions=1, model=None,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.train_transform = costume_transform
        self.repetitions = repetitions
        self.cifar10_path = cifar_path
        self.device = device
        self.model = model
        self.model.to(device)
        print(f"This is {type(self.model).__name__}")
        self.model.apply(init_weights)
        self.model.eval()

        

        self.images_matrix = None  # 初始化为 None
        self.L_12_betti_numbers_list = self.get_betti_number()
        self.feature2save = self.betti_number_feature(chose='L1')
        self.save_stats_to_file(file_path=save_file_path)
        self.feature2save = self.betti_number_feature(chose='L2')
        self.save_stats_to_file(file_path=save_file_path ,chose="L2")

    def images_to_matrix_lists(self):
        transform = self.train_transform
        # Download CIFAR-10 dataset
        trainset = torchvision.datasets.CIFAR10(root=self.cifar10_path, train=True, download=True,transform=self.train_transform)
        
        # 创建一个包含前1000个样本的子集
        subset_indices = range(1000)
        subset_dataset = Subset(trainset, subset_indices)
        # ----------------------------------这里其实不是很好，但是先这样选吧，后期应该做出修正--------------------------------

        self.images_matrix_lists = []  # Storing multiple image vector matrices

        for _ in range(self.repetitions):
            image_vectors = []  # Storing image vectors for each iteration
            for i in range(len(subset_dataset)):
                image, _ = subset_dataset[i]  # Get image and label (which is not used in this case)
                # print(type(image), "-"*10, image.shape)
                image = image.unsqueeze(0).to(self.device)
                image = self.model(image)[-1]
                image_vector = image.flatten()  # 转换为向量
                image_vector_array = image_vector.cpu().detach().numpy()
                image_vectors.append(image_vector_array)

            image_matrix = np.vstack(image_vectors)  # Stack image vectors to form a matrix
            self.images_matrix_lists.append(image_matrix)  # Append the matrix to the list
        return self.images_matrix_lists # 这个list中的每一个元素，矩阵，都代表了整个数据集
    
    def img_matrix2distance_matrix(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images_matrix_list = self.images_to_matrix_lists()  # 转换图像到矩阵列表
        self.all_l2_distances = []
        self.all_l1_distances = []

        for imgmatrix in images_matrix_list:
            l2_distances = vec_dis(data_matrix=torch.from_numpy(imgmatrix).to(device), distance="l2")
            l1_distances = vec_dis(data_matrix=torch.from_numpy(imgmatrix).to(device), distance="l1")
            self.all_l2_distances.append(l2_distances)
            self.all_l1_distances.append(l1_distances)

        return self.all_l1_distances, self.all_l2_distances


    def get_betti_number(self):
           
        self.l1_betti_number_list = []  # 吸收每一个特征图的的距离矩阵的betti numer
        self.l2_betti_number_list = []
        for l1_distance_matrix, l2_distance_matrix in zip(*self.img_matrix2distance_matrix()):    # 运行函数得到了L1，L2距离矩阵list
            # 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
            if 'rpp_py' in globals():
                d1 = rpp_py("--format distance --dim 1", l1_distance_matrix)
                d2 = rpp_py("--format distance --dim 1", l2_distance_matrix)
            else:

                d1 = ripser(l1_distance_matrix, maxdim=1, distance_matrix=True)
                d1 = d1["dgms"]
                d2 = ripser(l2_distance_matrix, maxdim=1, distance_matrix=True)
                d2 = d2["dgms"]
            d1 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix) for matrix in d1]  # 使用推导式，betti number中的那些无限大的采用最大值代替
            normalized_d1 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix / np.nanmax(matrix[np.isfinite(matrix)])) for matrix in d1]  # 实现betti number 层面上的归一化

            d2 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix) for matrix in d2]  # 使用推导式，betti number中的那些无限大的采用最大值代替
            normalized_d2 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix / np.nanmax(matrix[np.isfinite(matrix)])) for matrix in d2]  # 实现betti number 层面上的归一化

            self.l1_betti_number_list.append(normalized_d1)
            self.l2_betti_number_list.append(normalized_d2)
        return {"L1_betti_number_list":self.l1_betti_number_list, "L2_betti_number_list":self.l2_betti_number_list}    # 这个list中的每一个元素，矩阵，都代表了一个数据集

    def betti_number_feature(self, chose="L1"):
        if chose == "L1":
            betti_number_list = self.l1_betti_number_list
        elif chose == "L2":
            betti_number_list = self.l2_betti_number_list
        all_bars_survive_time_sum_list = []
        death_len_list = []

        for betti_number_matrix in betti_number_list:
            
            betti_number_matrix = betti_number_matrix[0]
            # print(betti_number_matrix)
            # 现在只关心all_bars_survive_time_sum和death_len
            all_bars_survive_time_sum = np.sum(betti_number_matrix[:, 1] - betti_number_matrix[:, 0])
            birth_len, death_len = calculate_edge_length(betti_number_matrix)

            all_bars_survive_time_sum_list.append(all_bars_survive_time_sum)
            death_len_list.append(death_len)

        results = {}
        results["all_bars_survive_time_sum"] = (statistics.mean(all_bars_survive_time_sum_list), statistics.stdev(all_bars_survive_time_sum_list))  # 存储均值和标准差为元组形式
        results["death_len"] = (statistics.mean(death_len_list), statistics.stdev(death_len_list))
        return results

    
    def save_stats_to_file(self, file_path, chose="L1"):
        # self.BOF_mean_stddev = self.calculate_stats()
        betti_features_path = os.path.join(file_path, f'{chose}_betti_features.pkl')
        # folder_path = os.path.dirname(file_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        with open(betti_features_path, 'wb') as f:
                pickle.dump(self.feature2save, f)

class CompareNetTDA():
    # 这里是为了比较不同的增强强度下的某一种model的表现力

    def __init__(self,
                file_path: str,
                target_pkl: str,
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
        save_path = os.path.join(self.folder_path, f'TDA_Data_{self.target_pkl.split(".")[0]}.png')
        data = self.comb_BOF

        # 创建4个子图的大图布局
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))  # 2个子图

        # 遍历每个子图的索引和对应的键
        keys = ['all_bars_survive_time_sum', 'death_len']  # 四个键
        for idx, subkey in enumerate(keys):
            # 提取每个子图的数据
            if aug_name == 'scale':
                values = [item[subkey][0] for item in data[::-1]]  # 提取mean值
                errors = [item[subkey][1] for item in data[::-1]]  # 提取标准差
            else:
                values = [item[subkey][0] for item in data]  # 提取mean值
                errors = [item[subkey][1] for item in data]  # 提取标准差

            # 绘制子图带误差棒
            axs[idx].errorbar(range(len(values)), values, yerr=errors, linestyle=':', marker='o', markersize=4)
            axs[idx].set_title(subkey)

        # 调整布局并保存图像
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()