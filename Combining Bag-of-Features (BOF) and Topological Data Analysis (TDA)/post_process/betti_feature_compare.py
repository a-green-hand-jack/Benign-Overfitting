# 这个模块就是为了处理那些pre process 得到数据，然后绘制

# 首先需要的是得到pkl路径下的文件
# 然后根据传入的参数得到我感兴趣的那部分
import pickle
from typing import Any, List, Union, Dict
from pprint import pprint
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示

# %% 加载目标pkl文件

class GetFeatureCared():
    # 得到的是某一个pkl文件中关心的特征

    def __init__(self,
                file_path: str,
                l_distance: str = 'L2',
                feature2get: str = 'all_bars_survive_time_sum',
                betti_number_dim: str = '1th',
                layer_care: bool = False,
                ) -> None:
        # pass
        self.l_distance = l_distance
        self.feature2get = feature2get
        self.betti_number_dim = betti_number_dim
        self.layer_care = layer_care

        self.betti_dict = self.try_load_pkl(file_path)
        self.feature_cared = self.care_betti_feature()
        
    def try_load_pkl(self, file_path: str) -> Union[Any, None]:
        """
        尝试加载一个Pickle文件。

        Args:
        - file_path (str): Pickle文件的路径

        Returns:
        - Union[Any, None]: 返回加载的数据，如果出现错误则返回None
        """
        # print(file_path)
        try:
            # 使用绝对路径加载 Pickle 文件
            with open(file_path, 'rb') as file:
                data = pickle.load(file)    # 这里的文件的路径最好使用绝对路径，不然打不开
                # print(f"{file_path} Pickle file loaded successfully.\n")
                # 这里可以添加你想要做的事情，比如打印数据内容
                # pprint(data)
                return data
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")
        except Exception as e:
            print("An error occurred:", e)


    def care_betti_feature(self) -> List[List[int]]:
        # 得到了某一个betti feature dict 中关心的特征

        feature_cared: List[List[int]] = []
        for key, value in self.betti_dict.items():
            if self.l_distance in key:
                # 确定我需要观察的是哪一种距离的定义
                # print(f"Key: {key}, Value: {value}")
                for sub_index, sub_value in enumerate(value[self.feature2get]):
                    # 确定我需要关注的特征
                    if self.betti_number_dim in str(sub_value.keys()):
                        # print(f"{sub_value}")
                        # 确定我需要关注的特征的betti dim 
                        feature_cared.append(list(sub_value.values()))

                if self.layer_care:
                    feature_cared = [value[0] for value in feature_cared]
                else:
                    feature_cared = [feature_cared[0][0], feature_cared[-1][0]]  
        # print(feature_cared)
        return feature_cared

class CompareFeatureCared:
    """
    A class to compare and combine features from multiple PKL files.

    Attributes:
    - folder_path (str): The folder path where PKL files are located.
    - target_pkl (str): The target PKL filename to compare against.
    - l_distance (str): The distance metric used for comparison (default: 'L2').
    - feature2get (str): The specific feature to extract from PKL files (default: 'all_bars_survive_time_sum').
    - betti_number_dim (str): The betti number dimension to consider (default: '1th').
    - layer_care (bool): Flag to indicate whether to care about layers (default: False).
    - matching_paths (list): List of matching PKL file paths.
    - comb_features (list): Combined features from matching PKL files.
    - comb_features_matrix (np.array or None): Combined features as a NumPy array.

    target_pkl:一般就是betti feature的保存的pkl文件;
    l_distance:L_1 or L_2;
    feature2get:bar_number or all_bars_survive_time_sum or max_epsilon_bar_number or death_len or max_death
    betti_number_dim:1th or 0th
    layer_care = False of True
    folder_path example: 
    """

    def __init__(self,
                 folder_path,
                 target_pkl='betti_features.pkl',
                 l_distance='L2',
                 feature2get='all_bars_survive_time_sum',
                 betti_number_dim='1th',
                 layer_care=False,
                 aug_type='angle') -> None:
        """
        Initialize CompareFeatureCared with specified parameters.
        """
        self.folder_path = folder_path
        self.target_pkl = target_pkl
        self.l_distance = l_distance
        self.feature2get = feature2get
        self.betti_number_dim = betti_number_dim
        self.layer_care = layer_care

        self.matching_paths = []
        self.comb_features = []
        self.comb_features_matrix = None

        self.find_matching_pkls()
        self.comb_feature_form_pkls()
        self.draw_betti(aug_type)

    # Other methods and functionalities...


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
        
        self.matching_paths = sorted_paths = sorted(self.matching_paths, key=get_number_from_path)

    def comb_feature_form_pkls(self):

        for pkl_path in self.matching_paths:
            temp_get_feature = GetFeatureCared(
                                            file_path=pkl_path,
                                            l_distance=self.l_distance,
                                            feature2get=self.feature2get,
                                            betti_number_dim=self.betti_number_dim,
                                            layer_care=self.layer_care
                                            )
            # print(temp_get_feature.feature_cared)
            self.comb_features.append(temp_get_feature.feature_cared)
        self.comb_features_matrix = np.array(self.comb_features)


    def draw_betti(self,aug_type):
        # 提取数据
        data = self.comb_features
        # save_path = os.join(self.folder_path, f'{self.l_distance}_{self.feature2get}_{self.betti_number_dim}_{self.layer_care}.png')
        save_path = os.path.join(self.folder_path, f'{self.l_distance}_{self.feature2get}_{self.betti_number_dim}_{self.layer_care}.png')
        
        # 获取所有类别
        categories = set(point[0] for sublist in data for point in sublist)
        colors = ['blue', 'red', 'green', 'orange', 'purple','cyan','yellow','lime','gold']  # 定义一些颜色
        markers = ['o', 's', '^', 'D', 'x','h', 'H', 'p', 'P', '8', '<', '>']  # 定义一些标记

        # 绘制曲线
        
        for idx, cat in enumerate(categories):
            x_values = [point[1] for sublist in data for point in sublist if point[0] == cat]
            y_values = [point[1] for sublist in data for point in sublist if point[0] == cat]
            if aug_type == "scale":
                x_axis = np.arange(len(x_values)) / 10 + 0.1
            elif aug_type == "angle":
                x_axis = np.arange(len(x_values))
                
            # plt.scatter(x_axis, y_values, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=str(cat))
            plt.plot(x_axis, y_values, linestyle=':', linewidth=1, markersize=5, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=str(cat))

        plt.xlabel('augmentation')
        plt.ylabel('feature')
        plt.title(f'{self.l_distance}_{self.feature2get}_{self.betti_number_dim}_{self.layer_care}')
        plt.legend()
        plt.savefig(save_path)
        plt.show()
        plt.close()
    # def draw_betti(self, aug_type):
    #     self.layer_care
    #     data = self.comb_features
    #     save_path = os.path.join(self.folder_path, f'{self.l_distance}_{self.feature2get}_{self.betti_number_dim}_{self.layer_care}.png')

    #     categories = set(point[0] for sublist in data for point in sublist)
    #     colors = ['blue', 'red', 'green', 'orange', 'purple']
    #     markers = ['o', 's', '^', 'D', 'x']

    #     # 绘制曲线
    #     y_values_list = []
    #     fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 创建一个1行2列的子图布局

    #     for idx, cat in enumerate(categories):
    #         x_values = [point[1] for sublist in data for point in sublist if point[0] == cat]
    #         y_values = [point[1] for sublist in data for point in sublist if point[0] == cat]
    #         y_values_list.append(y_values)

    #     if aug_type == "scale":
    #         x_axis = np.arange(len(x_values)) / 10 + 0.1
    #     elif aug_type == "angle":
    #         x_axis = np.arange(len(x_values))

    #     # 在第一个子图中绘制折线图
    #     axs[0].plot(x_axis, y_values_list[0], color=colors[0 % len(colors)], linestyle=':', marker=markers[0 % len(markers)], label='0', linewidth=0.5, markersize=5)

    #     # 在第二个子图中绘制折线图
    #     axs[1].plot(x_axis, y_values_list[1], color=colors[1 % len(colors)], linestyle=':', marker=markers[1 % len(markers)], label='3', linewidth=0.5, markersize=5)

    #     axs[0].set_xlabel('augmentation')
    #     axs[0].set_ylabel('feature')
    #     axs[0].set_title(f'{self.l_distance}_{self.feature2get}_{self.betti_number_dim}_{self.layer_care}')
    #     axs[0].legend()

    #     axs[1].set_xlabel('augmentation')
    #     axs[1].set_ylabel('feature')
    #     axs[1].set_title(f'{self.l_distance}_{self.feature2get}_{self.betti_number_dim}_{self.layer_care}')
    #     axs[1].legend()

    #     plt.tight_layout()
    #     plt.savefig(save_path)
    #     plt.show()
    #     plt.close()
