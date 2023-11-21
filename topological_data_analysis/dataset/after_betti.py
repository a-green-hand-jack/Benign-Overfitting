import numpy as np
import pickle
import os
import re
from typing import Tuple

from dataset.get_betti_number import check_folder_integrity


def try_parse_float(s):
    """
    尝试将字符串转换为浮点数。如果成功，返回浮点数；否则，返回原始字符串。

    参数：
    - s：要尝试转换的字符串。

    返回：
    成功转换为浮点数的值，或者原始字符串。
    """
    try:
        return float(s)
    except ValueError:
        return s

def custom_sort(filename):
    """
    自定义排序函数，按照文件名中的数字部分进行排序。

    参数：
    - filename：要排序的文件名字符串。

    返回：
    包含数字部分的列表，用于排序。
    """
    return [try_parse_float(c) for c in re.split('(\d+.\d+|\d+)', filename)]

def after_get_bars(base_path="distance/angle/LeNet/"):
    """
    从给定的文件夹中操作，对所有的文件夹中的betti_bar进行操作，得到对应的几个观察角度：
    - bar的数量
    - bars的存活时间
    - 最好的ε
    - 边缘的长度

    参数：
    - base_path：基本路径，即包含所有文件夹的文件夹的路径，默认为"distance/angle/LeNet/"。
    """
    files = os.listdir(base_path)
    files.sort(key=custom_sort)
    for path in files:
        # print(path)
        full_path = os.path.join(base_path, path)
        print(full_path)
        check_result,true_png,true_pkl = check_folder_integrity(full_path, min_png=6, min_pkl=1)
        if check_result:
            print(f"观察{full_path}文件夹,里面有{true_png}张图片、{true_pkl}份betti_number.pkl数据。可以计算after_betti。")
            # full_path = os.path.join(base_path, path)
            test_dict = convert_encoding(full_path, "betti_number.pkl")
            get_all_for_betti(test_dict, full_path)
        else:
            print(f"观察{full_path}文件夹,里面有{true_png}张图片、{true_pkl}份betti_number.pkl数据。期望有6张图片，1份betti_number.pkl文件")
            pass



def convert_encoding(folder_path, name):
    '''
    从给定的文件夹中读取.pkl文件并重新整理数据。
    重要的是对文件夹的重新处理

    参数：
    - folder_path：文件夹路径。
    - name：文件名。

    返回：
    - 重新整理后的字典数据。
    '''
    file_path = os.path.join(folder_path, name)  
    f_read = open(file_path, "rb") 
    betti_data = pickle.load(f_read)   
    # print(betti_data)  
    my_dict = {"L1-B0":betti_data["BD-L1"][0], "L1-B1":betti_data["BD-L1"][1], "L2-B0":betti_data["BD-L2"][0],"L2-B1":betti_data["BD-L2"][1]}
    # print(my_dict)
    return my_dict


def count_samples(matrix, number):
    '''
    计算给定数字落在矩阵中样本的最大最小值之间的数量。

    参数：
    - matrix：一个包含样本区间的矩阵，每一行表示一个样本区间，包含最小值和最大值。
    - number：给定的数字。

    返回：
    - 落在给定数字范围内的样本数量。
    '''
    # 获取矩阵中每一行的最大值和最小值
    max_values = matrix[:, 1]
    min_values = matrix[:, 0]

    # 使用条件判断计算落在范围内的样本数目
    count = np.sum((number >= min_values) & (number <= max_values))

    return count



def get_min_max_columns(matrix):
    '''
    获取矩阵的第一列的最小值和第二列的最大值。

    参数：
    - matrix：待处理的矩阵。

    返回：
    - min_first_column：第一列的最小值。
    - max_second_column：第二列的最大值。
    '''
    # 将无穷大值替换为最大实数
    finite_matrix = np.where(np.isfinite(matrix), matrix, np.max(matrix[np.isfinite(matrix)]))

    # 计算第一列的最小值
    min_first_column = np.min(finite_matrix[:, 0])

    # 计算第二列的最大值
    max_second_column = np.max(finite_matrix[:, 1])

    return min_first_column, max_second_column, finite_matrix




def calculate_edge_length(matrix: np.ndarray) -> Tuple[float, float]:
    '''
    计算曲线的近似长度。
    使用离散曲线的长度计算方法：通过对相邻顶点之间的距离进行累加来计算曲线的长度。
    这种方法可以适用于任意数量的顶点，而不需要人为规定任意两个顶点之间的距离。

    参数：
    - matrix：包含曲线顶点的矩阵，每一行表示一个顶点，包含左边界和右边界。

    返回：
    - left_line：左边界曲线的近似长度。
    - right_line：右边界曲线的近似长度。
    '''
    # 获取左边界和右边界的顶点坐标
    left_vector = matrix[:, 0]
    right_vector = matrix[:, 1]

    # 计算左边界的顶点长度
    left_distance = np.sqrt(np.sum(np.diff(left_vector, axis=0) ** 2, axis=1))
    left_line = np.sum(left_distance)

    # 计算右边界的顶点长度
    right_distance = np.sqrt(np.sum(np.diff(right_vector, axis=0) ** 2, axis=1))
    right_line = np.sum(right_distance)

    return left_line, right_line



def save_dict(dictionary, file_path):
    '''
    将字典保存到文件中。

    参数：
    - dictionary：要保存的字典。
    - file_path：文件路径。
    '''
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)


def get_all_for_betti(bar_dict=None, save_root=None):

    '''
    对保存好的betti_number.pkl文件进行一系列操作，得到一些观察角度。

    参数：
    - bar_dict：包含bar数据的字典（由convert_encoding函数生成）。
    - save_root：保存结果的根目录路径。

    返回：
    - bar_number_dict：包含bar数目的字典。
    - all_bars_survive_time_sum_dict：包含所有bar存活时间之和的字典。
    - max_betti_number_epsilon_dict：包含最多bar对应的epsilon和bar数目的字典。
    - birth_len_dict：包含生线长度的字典。
    - death_len_dict：包含死线长度的字典。

    保存：
    - 按照save_root路径将上述5个字典打包成一个字典之后，保存在save_root路径下的after_betti_number.pkl文件
    '''

    bar_number_dict = {}
    all_bars_survive_time_sum_dict = {}
    max_betti_number_epsilon_dict = {}
    birth_len_dict = {}
    death_len_dict = {}
    for key, value in bar_dict.items():
        # print(key, value)
        min_first_colum, max_second_column,matrix = get_min_max_columns(value)
        # if value[-1, -1] == float("inf"):

        #     matrix = value[:-1, :]
        # else:
        #     matrix = value

        # 第一，计算bar的数目
        bar_number_key = f"bar_number_{key}"
        bar_number_dict[bar_number_key] = matrix.shape[0]

        # 第二，计算所有bar的存活时间之和
        all_bars_survive_time_sum_key = f"all_bars_survive_time_sum_{key}"
        all_bars_survive_time_sum_dict[all_bars_survive_time_sum_key] = np.sum(matrix[:,1] - matrix[:,0])

        # 第三，计算最多的bar对应的ε和bar，b1(ε)_max,ε

        max_cont = 0
        max_epsilon = 0

        # min_first_colum, max_second_column = get_min_max_columns(matrix)
        min_first_colum, max_second_column = int(min_first_colum), int(max_second_column)

        for epsilon in range(min_first_colum, 1+max_second_column):
            counter = count_samples(matrix, epsilon)
            if counter >= max_cont:
                max_cont = counter
                max_epsilon = epsilon
        
        max_betti_number_epsilon_key = f"max_betti_number_epsilon_{key}"
        max_betti_number_epsilon_dict[max_betti_number_epsilon_key] = (max_epsilon, max_cont)
        
        # 第四，计算生/死线的长度

        birth_len, death_len = calculate_edge_length(matrix)
        birth_key = f"birth_len_{key}"
        death_key = f"death_len_{key}"
        birth_len_dict[birth_key] = birth_len
        death_len_dict[death_key] = death_len

        if save_root is not None:

            root = f"{save_root}/after_betti_number.pkl"
            # np.save(root, {d1_key:d1["dgms"],d2_key:d2["dgms"]}) # 注意带上后缀名
            
            # 保存字典到文件
            dict_my =  {"bar_number_dict":bar_number_dict, "all_bars_survive_time_sum_dict":all_bars_survive_time_sum_dict, "max_betti_number_epsilon_dict":max_betti_number_epsilon_dict, "birth_len_dict":birth_len_dict, "death_len_dict":death_len_dict}

            save_dict(dict_my, root)
        
    return bar_number_dict, all_bars_survive_time_sum_dict, max_betti_number_epsilon_dict, birth_len_dict, death_len_dict


