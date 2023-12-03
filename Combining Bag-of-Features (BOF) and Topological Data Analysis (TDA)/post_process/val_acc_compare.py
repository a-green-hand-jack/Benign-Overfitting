
import pickle
from typing import Any, List, Union, Dict
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示

class CompareValAcc():
    # 这里是为了比较不同的增强强度下的某一种model的表现力

    def __init__(self,
                file_path: str,
                target_pkl: str,
                net_name: str='MLP',
                aug_name: str='angle'


                ) -> None:
        self.folder_path = file_path
        self.target_pkl = target_pkl

        self.matching_paths = []
        self.find_matching_pkls()

        self.comb_acc = []
        self.comb_acc_matrix = None
        self.compare_acc()


        self.draw_acc(net_name=net_name, aug_name=aug_name)


        
        
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

    def compare_acc(self):
        for pkl_path in self.matching_paths:
            temp_get_acc = self.try_load_pkl(file_path=pkl_path)
            # print(temp_get_acc.feature_cared)
            value = list(temp_get_acc.values())[0]
            # print('{}'.format(value))
            self.comb_acc.append(value)
            # print(temp_get_acc)
        self.comb_acc_matrix = np.array(self.comb_acc)

    def draw_acc(self, net_name, aug_name):
        values = self.comb_acc_matrix
        save_path = os.path.join(self.folder_path, f'test_acc_{net_name}_{aug_name}.png')

        # 创建x轴数据（例如，可以从0开始的整数索引）
        x_values = np.arange(len(values))

        # 绘制折线图
        plt.figure(figsize=(8, 6))

        # 绘制线条
        plt.plot(x_values, values, linestyle=':', linewidth=1, marker='o', markersize=5)

        # 设置图表标题和坐标轴标签
        plt.title(f'{net_name}_{aug_name}')
        plt.xlabel('augmentation')
        plt.ylabel('test acc')

        # 显示图例
        plt.legend(['test_acc'], loc='upper right')

        # 显示图表
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()





class CompareBOF():
    # 这里是为了比较不同的增强强度下的某一种model的表现力

    def __init__(self,
                file_path: str,
                target_pkl: str,
                net_name: str='MLP',
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

        self.draw_BOF(net_name=net_name, aug_name=aug_name)
        


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
        # 获取子图的数量
        
        save_path = os.path.join(self.folder_path, f'BOF_{net_name}_{aug_name}.png')
        data = self.comb_BOF
        print(data)
        num_subplots = len(data[0][f'{net_name}+{aug_name}'])

        # 创建大图布局
        fig, axs = plt.subplots(1, num_subplots, figsize=(15, 5))

        # 遍历每个子图的索引和子字典
        for idx, subkey in enumerate(data[0][f'{net_name}+{aug_name}']):
            # 提取每个子图的数据
            values = [item[f'{net_name}+{aug_name}'][subkey] for item in data]

            # 绘制子图，设置点划线和大点的样式
            axs[idx].plot(range(len(values)), values, linestyle=':', marker='o', markersize=4)
            axs[idx].set_title(subkey)

        # 调整布局
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()
        










