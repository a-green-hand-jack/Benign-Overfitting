import pickle
import os
from dataset.after_betti import custom_sort
from collections import OrderedDict
import matplotlib.pyplot as plt
from dataset.get_betti_number import check_folder_integrity
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示



"""_summary_

所谓的transform_AB就是对after betti number 进行处理，目的是为了整合通过after betti number 部分得到的各个分散的betti number 的描述
会在每一个文件夹【同一中增强类型下的不同的网络和数据】产生一个`compare_different_bitte_norm_in_same_augmentation.pkl`文件
这个文件有着这样的结构：
    .\distance\angle\LeNet\compare_different_bitte_norm_in_same_augmentation.pkl
        {'L1-B0': {'all_bars_survive_time_sum': {'angle': 130.09174275398254},
                'bar_number': {'angle': 99},
                'birth_len': {'angle': 0.98},
                'death_len': {'angle': 2.5719317884935906},
                'max_betti_number_epsilon': {'angle': (0, 99)}},
        'L1-B1': {'all_bars_survive_time_sum': {'angle': 8.392859816551208},
                'bar_number': {'angle': 63},
                'birth_len': {'angle': 1.9763853823194673},
                'death_len': {'angle': 8.669621438271115},
                'max_betti_number_epsilon': {'angle': (2, 3)}},
        'L2-B0': {'all_bars_survive_time_sum': {'angle': 51.49522919440642},
                'bar_number': {'angle': 99},
                'birth_len': {'angle': 0.9894136528412905},
                'death_len': {'angle': 1.5387326987190524},
                'max_betti_number_epsilon': {'angle': (0, 80)}},
        'L2-B1': {'all_bars_survive_time_sum': {'angle': 2.318040370941162},
                'bar_number': {'angle': 49},
                'birth_len': {'angle': 0.9068361532976893},
                'death_len': {'angle': 2.6244334764079698},
                'max_betti_number_epsilon': {'angle': (1, 2)}}}

"""

def process_betti_pickle(path):
    """
    处理 Betti pickle 文件并转换为 DataFrame。

    参数：
    - path (str): Betti pickle 文件的文件路径。

    返回：
    - 一个二维的字典。

    Example usage:
        file_path = "./distance/angle/data/0/after_betti_number.pkl"
        result_df = process_betti_pickle(file_path)
        pprint(result_df)

    
    Example result:
        {   'all_bars_survive_time_sum': {'L1-B0': 256471.4666748047,
                               'L1-B1': 2114.12158203125,
                               'L2-B0': 5862.379766043276,
                               'L2-B1': 33.46942901611328},

            'bar_number': {'L1-B0': 99, 'L1-B1': 21, 'L2-B0': 99, 'L2-B1': 19},

            'birth_len': {'L1-B0': 0.98,
                        'L1-B1': 1299.0995394369627,
                        'L2-B0': 1.7042854619485461,
                        'L2-B1': 25.126771145885744},

            'death_len': {'L1-B0': 2305.5485960567385,
                        'L1-B1': 2472.0125763396277,
                        'L2-B0': 47.714379134280705,
                        'L2-B1': 41.14415296480198},

            'max_betti_number_epsilon': {'L1-B0': (1575, 99),
                                        'L1-B1': (2293, 6),
                                        'L2-B0': (38, 99),
                                        'L2-B1': (68, 4)}
                                        }

    """
    # 读取 Pickle 文件
    # if os.path.isdir(path):
    try:
        with open(path, 'rb') as file:
            betti_data = pickle.load(file)
        # 处理第一层键
    except FileNotFoundError:
        pass

    # 处理第一层键
    modified_keys = {}
    for key, value in betti_data.items():
        if key.endswith("_dict"):
            new_key = key.rsplit("_", 1)[0]  # 去掉末尾的 "_dict"
            modified_keys[new_key] = {}

            # 处理第二层键
            for sub_key, sub_value in value.items():
                new_sub_key = sub_key.split("_")[-1]  # 保留 "L1-B0" 部分
                modified_keys[new_key][new_sub_key] = sub_value

    return modified_keys

def compare_after_betti_in_same_augmentation(base_path="distance/angle/LeNet/"):
    """
    从给定的文件夹中操作，对所有的文件夹中的betti_bar进行操作，得到对应的几个观察角度：
    - bar的数量
    - bars的存活时间
    - 最好的ε
    - 边缘的长度

    这样将有利于观察不同的距离的定义下的四个不同的考察指标，在同一种增强的不同增强强度下的变化。
    参数：
    - base_path：基本路径，即包含所有文件夹的文件夹的路径，默认为"distance/angle/LeNet/"。

    返回：
    - 经过整理的，其实是转置的，一个三维字典，有利于观察和后续计算
    - 还需要把得到这个字典保存在这个文件夹下，以便未来的访问

    案例：
    -   k = after_get_bars("./distance/Mixup/data/")
        pprint(k)

        结果：
            {'L1-B0': {'all_bars_survive_time_sum': {'Mixup/data/0.0': 256471.4666748047,
                                         'Mixup/data/0.05': 254161.13317871094,
                                         'Mixup/data/0.1': 211884.4747314453,
                                         'Mixup/data/0.15000000000000002': 255776.95092773438,
                                         'Mixup/data/0.2': 255328.78076171875,
                                         'Mixup/data/0.25': 254117.20239257812,
                                         'Mixup/data/0.30000000000000004': 241949.62475585938,
                                         'Mixup/data/0.35000000000000003': 208344.15795898438,
                                         'Mixup/data/0.4': 255801.08618164062,
                                         'Mixup/data/0.45': 223151.09533691406,}}
    """
    def transpose_dict(d):
        transposed_dict = {}
        for key1, value1 in d.items():
            for key2, value2 in value1.items():
                for key3, value3 in value2.items():
                    if key3 not in transposed_dict:
                        transposed_dict[key3] = {}
                    if key2 not in transposed_dict[key3]:
                        transposed_dict[key3][key2] = {}
                    transposed_dict[key3][key2][key1] = value3
        return transposed_dict
    

    files = os.listdir(base_path)
    files.sort(key=custom_sort)
    # print(files)
    result_3d_dfs = {}
    for path in files:
        # print(path)
        # print(path)
        full_path = os.path.join(base_path, path)
        # print(full_path)
        
        # 判断路径是否是满足要求的文件夹
        check_result, _, _ = check_folder_integrity(folder_path=full_path, min_png=6, min_pkl=2)
        if check_result:
            file_path = os.path.join(full_path, "after_betti_number.pkl")  
            # print(file_path)
            file_name = file_path.split("\\", 1)[1].split("\\after_betti_number.pkl")[0]
            # print(file_name)
            result_df = process_betti_pickle(file_path)

            result_3d_dfs[file_name] = result_df


    # result_3d_df = pd.concat(result_3d_dfs, keys=result_3d_dfs.keys(), names=['file_name'], axis=0)
    result_3d_dfs = OrderedDict(result_3d_dfs)
    # pprint(result_3d_dfs)
    better_transpose_dict = transpose_dict(result_3d_dfs)
    better_transpose_dict = OrderedDict(better_transpose_dict)
    # pprint(better_transpose_dict)

    save_root = base_path

    root = f"{save_root}/compare_different_bitte_norm_in_same_augmentation.pkl"
    # print(root)
    # np.save(root, {d1_key:d1["dgms"],d2_key:d2["dgms"]}) # 注意带上后缀名
            
            # 保存字典到文件
    with open(root, 'wb') as file:
        pickle.dump(better_transpose_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

        # print("00000000")
        # print(f"{better_transpose_dict} is saved in {root}")
        # pprint(better_transpose_dict)

    return None

def get_all_cabs(base_path):
    parent_files = os.listdir(base_path)
    parent_files.sort(key=custom_sort)
    result_3d_dfs = {}
    for child in parent_files:
        child_path = os.path.join(base_path, child)
        # 判断路径是否是一个文件夹
        if os.path.isdir(child_path):
            # print(child_path)
            child_files = os.listdir(child_path)
            child_files.sort(key=custom_sort)
            for grandchild in child_files:
                grandchild_path = os.path.join(child_path, grandchild)
                if os.path.isdir(grandchild_path):
                    print(grandchild_path)
                    _ = compare_after_betti_in_same_augmentation(grandchild_path)
                    # print(grandchild_path)

    return None

def show_intraspecific_differences(grandchild_path):
    """
    Visualize intraspecific differences and save the plots.

    Parameters:
    - grandchild_path (str): The path to the folder containing data.

    Returns:
    None
    """

    # Create 'intraspecific_differences' folder if not exists
    save_folder = os.path.join(grandchild_path, "intraspecific_differences")
    os.makedirs(save_folder, exist_ok=True)

    after_betti_data_path = os.path.join(grandchild_path, "compare_different_bitte_norm_in_same_augmentation.pkl")
    with open(after_betti_data_path, 'rb') as file:
        your_dict = pickle.load(file)

        # 遍历第一层
        for l1_key, l1_value in your_dict.items():
            # 遍历第二层
            for l2_key, l2_value in l1_value.items():
                # 设置图表标题
                plt.title(f'采用的距离种类和betti阶数是{l1_key}')

                # 遍历第三层并绘制散点图
                for l3_key, l3_value in l2_value.items():
                    x_value = float(l3_key.split('\\')[-1])  # 提取横坐标的值
                    if isinstance(l3_value, tuple):
                        # 如果值是元组，分别提取两个值
                        epsilon, betti_number = l3_value
                        plt.scatter(x_value, epsilon, label=f'{l3_key} - ε', color='blue')
                        plt.scatter(x_value, betti_number, label=f'{l3_key} - betti number', color='red')
                    else:
                        # 否则，正常绘制
                        y_value = l3_value  # 提取纵坐标的值
                        plt.scatter(x_value, y_value, label=l3_key)

                # 设置图表标签
                plt.xlabel(f'{grandchild_path}下的增强的强度')
                plt.ylabel(f'{l2_key}')

                # 显示图例
                # plt.legend()

                # 保存图表
                save_filename = f'{l1_key}_{l2_key}.png'
                save_path = os.path.join(save_folder, save_filename)
                plt.savefig(save_path)

                # 显示图表
                # plt.show()
                plt.close()

def show_all_different(base_path):
    # 这里的目的是对种内差异实现可视化
    parent_files = os.listdir(base_path)
    parent_files.sort(key=custom_sort)
    result_3d_dfs = {}
    for child in parent_files:
        child_path = os.path.join(base_path, child)
        # 判断路径是否是一个文件夹
        if os.path.isdir(child_path):
            # print(child_path)
            child_files = os.listdir(child_path)
            child_files.sort(key=custom_sort)
            for grandchild in child_files:
                grandchild_path = os.path.join(child_path, grandchild)
                if os.path.isdir(grandchild_path):
                    print(grandchild_path)
                    show_intraspecific_differences(grandchild_path)
                #    print(grandchild_path)

if __name__ == '__main__':
    
    base_path = ".\\distance"
    get_all_cabs(base_path)
