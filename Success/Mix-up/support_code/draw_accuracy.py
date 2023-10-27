import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']


def draw_accuracy(input_dir="csv文件所在位置", output_dir="png文件输出的位置"):

    # 如果目录不存在，创建它
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # 列出包含数据的CSV文件
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # 计算行数和列数以排列子图
    num_plots = len(csv_files)
    num_rows = 3  # 3行
    num_cols = (num_plots + num_rows - 1) // num_rows  # 自动计算列数

    # 创建大图
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))

    # 循环处理每个CSV文件
    for i, csv_file in enumerate(csv_files):
        # 构建CSV文件的完整路径
        csv_path = os.path.join(input_dir, csv_file)

        # 读取CSV文件为DataFrame
        df = pd.read_csv(csv_path)

        # 获取横坐标和纵坐标数据
        x_data = df.iloc[:, 1]  # 第二列作为横坐标
        y_data = df.iloc[:, 2]  # 第三列作为纵坐标

        # 计算当前子图的位置
        row_idx = i // num_cols
        col_idx = i % num_cols

        # 绘制子图
        axs[row_idx, col_idx].plot(x_data, y_data)
        axs[row_idx, col_idx].set_title(f"{csv_file}")  # 使用文件名作为子图标题
        axs[row_idx, col_idx].set_xlabel(df.columns[1])  # 使用第二列列名作为横坐标轴标签
        axs[row_idx, col_idx].set_ylabel(df.columns[2])  # 使用第三列列名作为纵坐标轴标签
        axs[row_idx, col_idx].grid()

    # 调整布局
    plt.tight_layout()

    # 保存整个大图
    input_filename = os.path.join(output_dir, "combined_plot.png")
    plt.savefig(input_filename)
    # print(f"Combined plot saved as '{input_filename}'")

    # 显示图形
    # plt.show()

def draw_acc_change(input_dir="csv文件所在位置",  output_dir="png文件输出的位置"):
    """"
    输入通过tensorboard处下载的csv文件所在的文件夹,在同样的位置各个csv文件对应的图像
    """
    # 如果目录不存在，创建它
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # 列出包含数据的CSV文件
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # 循环处理每个CSV文件
    for i, csv_file in enumerate(csv_files):
        # 构建CSV文件的完整路径
        csv_path = os.path.join(input_dir, csv_file)

        # 读取CSV文件为DataFrame
        df = pd.read_csv(csv_path)

        # 获取横坐标和纵坐标数据
        x_data = df.iloc[:, 1]  # 第二列作为横坐标
        y_data = df.iloc[:, 2]  # 第三列作为纵坐标

        # 创建新的图形窗口
        plt.figure()

        # 绘制子图
        plt.scatter(x_data, y_data )
        plt.title(f"{csv_file}")  # 使用文件名作为子图标题
        plt.xlabel(df.columns[1])  # 使用第二列列名作为横坐标轴标签
        plt.ylabel(df.columns[2])  # 使用第三列列名作为纵坐标轴标签
        plt.grid()

        # 保存子图
        input_filename = os.path.join(output_dir, f"{csv_file.split('.csv')[0]}.png")  # 文件名与题目一致，去除扩展名并添加.png后缀
        plt.savefig(input_filename)
        # print(f"Plot saved as '{input_filename}'")


def best_valid_acc(input_dir="csv文件所在位置",  output_dir="png文件输出的位置", plot=False, use_text=True):

    # 创建字典用于保存最大值
    max_values = {}

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # 列出包含数据的CSV文件
    path_list = os.listdir(input_dir)
    path_list.sort(key=lambda x:int(x.split('-')[0]) )
    csv_files = [file for file in path_list if file.endswith('.csv')]
    
    # 循环处理每个CSV文件
    for i, csv_file in enumerate(csv_files):
        print(f"{csv_file}")
        scale_key = csv_file.split(".csv")[0]
        # 构建CSV文件的完整路径
        csv_path = os.path.join(input_dir, csv_file)

        # 读取CSV文件为DataFrame
        df = pd.read_csv(csv_path)

        # 获取横坐标和纵坐标数据
        # x_data = df.iloc[:, 4]  # 第二列作为横坐标
        x_data = range(len(df.iloc[:,4]))
        y_data = df.iloc[:, 4]  # 第三列作为纵坐标
         # 计算最大值并保存到字典
        max_value = y_data.max()
        # key = f"scale=0.{scale_index}"
        max_values[scale_key] = max_value

    # 打印保存的最大值字典
    print("最大值字典:")
    for key, value in max_values.items():
        print(f"{key}: {value}")

    # 绘制子图
    if plot:
        plt.plot(range(len(max_values.keys())), max_values.values(), marker="*")
    else:
        plt.scatter(range(len(max_values.keys())), max_values.values() )

    if use_text:
        for i, j in zip(range(len(max_values.keys())), max_values.values()):
            plt.text(i, j, f'({j})', fontsize=8, ha='center', va='bottom')
    plt.title("观察valid中最大准确率的变化")  # 使用文件名作为子图标题
    plt.ylabel("max_valid_accuracy")  #为横坐标轴标签
    plt.xlabel("scale")  # 使用第三列列名作为纵坐标轴标签
    plt.grid()

    # 保存子图
    input_filename = os.path.join(output_dir, "观察valid中最大准确率的变化.png")  # 文件名与题目一致，去除扩展名并添加.png后缀
    plt.savefig(input_filename)
    # print(f"Plot saved as '{input_filename}'")

def extract_file_names(directory, min_values, max_values):
    """
    从指定目录中提取符合指定范围的文件名列表。

    参数:
    directory (str): 包含文件的目录路径。
    min_values (tuple): 最小值的元组，如 (0.1000, 0.1000, 0.1000, 0.1000)。
    max_values (tuple): 最大值的元组，如 (0.1000, 0.1000, 0.1000, 1.0000)。

    返回:
    list: 符合范围条件的文件名列表。
    """
    file_names = os.listdir(directory)
    selected_file_names = []

    for file_name in file_names:
        # 从文件名中提取浮点数值，去除空格和其他字符
        values = tuple(map(float, file_name.strip('().csv').split(',')))

        # 检查每个值是否在范围内
        is_in_range = all(min_val <= val <= max_val for val, min_val, max_val in zip(values, min_values, max_values))

        if is_in_range:
            selected_file_names.append(file_name)

    return selected_file_names



def chose_best_valid_acc(input_dir, 
                         output_dir, 
                         csv_list=None, 
                         new_min_values=None, 
                         new_max_values=None,
                         png = None):
    """
    从指定目录中的一组CSV文件中提取最大准确率，并绘制观察valid中最大准确率的变化。

    参数:
    input_dir (str): 包含CSV文件的目录路径。
    output_dir (str): 保存生成图片的目录路径。
    csv_list (list): 需要处理的CSV文件名列表。如果为None，将处理目录中的所有CSV文件。
    min_values (tuple): 使用的最小值
    max_values (tuple): 使用的最大值

    使用示例:
    best_valid_acc(input_dir="csv文件所在位置", output_dir="png文件输出的位置", csv_list=["file1.csv", "file2.csv"],
                   min_values=(0.1, 0.1, 0.1, 0.1), max_values=(1.0, 1.0, 1.0, 1.0))
    
    参数input_dir指定了CSV文件所在的目录，output_dir指定了生成的PNG图片保存的目录。
    参数csv_list是一个可选参数，用于指定需要处理的CSV文件列表。如果不提供该参数，函数将处理目录中的所有CSV文件。
    函数会遍历每个CSV文件，提取最大准确率，并绘制一个观察valid中最大准确率的变化的折线图，最后保存在output_dir中。

    """
    # 列出包含数据的CSV文件
    csv_files = [file for file in os.listdir(input_dir) if file in csv_list]

    # 创建字典用于保存最大值
    max_values = {}

    # 循环处理每个CSV文件
    for i, csv_file in enumerate(csv_files):
        scale_key = csv_file.split(".csv")[0]
        # 构建CSV文件的完整路径
        csv_path = os.path.join(input_dir, csv_file)

        # 读取CSV文件为DataFrame
        df = pd.read_csv(csv_path)

        # 获取横坐标和纵坐标数据
        x_data = range(len(df.iloc[:, 4]))
        y_data = df.iloc[:, 4]  # 第三列作为纵坐标

        # 计算最大值并保存到字典
        max_value = y_data.max()
        max_values[scale_key] = max_value

    # 打印保存的最大值字典
    # print("最大值字典:")
    # for key, value in max_values.items():
    #     print(f"{key}: {value}")

    # 绘制子图
    plt.scatter(range(len(max_values.keys())), max_values.values())
    title = f"{new_min_values}-{new_max_values}"  # 根据传递的min_values和max_values构建标题
    if png == None:
        png = f"{title}.png"
    plt.title(title)  # 使用标题作为子图标题
    plt.ylabel("最佳准确率变化情况")  # 为横坐标轴标签
    plt.xlabel("ColorJitter")  # 使用第三列列名作为纵坐标轴标签
    plt.grid()

    # 保存子图
    input_filename = os.path.join(output_dir, png)  # 使用标题作为文件名，去除扩展名并添加.png后缀
    plt.savefig(input_filename)
    # 及时关闭
    plt.close()


def best_valid_acc_group_index(input_dir="csv文件所在位置", output_dir="png文件输出的位置", group_size=50, file_interval=5):
    """
    从指定目录中的一组CSV文件中提取最大准确率，并绘制观察valid中最大准确率的变化的子图，带有组和文件间隔索引。

    参数:
    input_dir (str): 包含CSV文件的目录路径。
    output_dir (str): 保存生成图片的目录路径。
    group_size (int): 每个子组中包含的CSV文件数。
    file_interval (int): 在每个子组中，每隔多少个文件处理一次。

    使用示例:
    best_valid_acc_group_index(input_dir="csv文件所在位置", output_dir="png文件输出的位置", group_size=50, file_interval=5)

    参数input_dir指定了CSV文件所在的目录，output_dir指定了生成的PNG图片保存的目录。
    参数group_size是每个子组中包含的CSV文件数，file_interval表示在每个子组中，每隔多少个文件处理一次。
    函数会遍历每个子组的CSV文件，提取最大准确率，并绘制带有组和文件间隔索引的子图，最后保存在output_dir中。

    """
    # 列出包含数据的CSV文件
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # 将文件名分组，每组包含 group_size 个文件
    grouped_files = [csv_files[i:i + group_size] for i in range(0, len(csv_files), group_size)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for group_index, group in enumerate(grouped_files):
        # 创建字典用于保存最大值
        max_values = {}

        # 循环处理每个CSV文件，间隔为file_interval
        for i in range(0, len(group), file_interval):
            csv_file = group[i]
            scale_key = csv_file.split(".csv")[0]
            # 构建CSV文件的完整路径
            csv_path = os.path.join(input_dir, csv_file)

            # 读取CSV文件为DataFrame
            df = pd.read_csv(csv_path)

            # 获取横坐标和纵坐标数据
            x_data = range(len(df.iloc[:, 4]))
            y_data = df.iloc[:, 4]
            
            # 计算最大值并保存到字典
            max_value = y_data.max()
            max_values[scale_key] = max_value

        # 绘制子图
        plt.figure()
        plt.scatter(range(len(max_values.keys())), max_values.values())
        plt.title(f"观察valid中最大准确率的变化 - Group {group_index + 1}- index {file_interval}")
        plt.ylabel("max_valid_accuracy")
        plt.xlabel("scale")
        plt.grid()

        # 保存子图
        input_filename = os.path.join(output_dir, f"Group{group_index + 1}_index{file_interval}.png")
        plt.savefig(input_filename)
        plt.close()  # 关闭图形，以便绘制下一个子图

def extract_max_values(input_dir="csv文件所在位置"):
    # 列出包含数据的CSV文件
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # 创建一个空的NumPy数组来存储最大值
    max_values = np.zeros(len(csv_files))

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # 循环处理每个CSV文件
    for i, csv_file in enumerate(csv_files):
        scale_key = csv_file.split(".csv")[0]
        # 构建CSV文件的完整路径
        csv_path = os.path.join(input_dir, csv_file)

        # 读取CSV文件为DataFrame
        df = pd.read_csv(csv_path)

        y_data = df.iloc[:, 4]  # 第三列作为纵坐标
        # 计算最大值并保存到NumPy数组
        max_value = y_data.max()
        max_values[i] = max_value

    return max_values

def plot_values_of_interest(max_values_matrix, indices_of_interest, output_dir="png文件输出的位置", plot=False):
    """
    绘制关心的值在 max_values_matrix 中的散点图并保存为PNG文件。

    参数:
    max_values_matrix (numpy.ndarray): 5x5x5x5 的 NumPy 矩阵，包含最大值数据。
    indices_of_interest (list): 关心的值的索引列表。
    output_dir (str, optional): 保存输出图像的目录路径。默认为 "png文件输出的位置"。

    返回:
    None
    """
    # 检查输出目录是否存在，如果不存在则创建
    title = indices_of_interest
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查 max_values_matrix 的类型和形状是否正确
    if not isinstance(max_values_matrix, np.ndarray) or max_values_matrix.shape != (5, 5, 5, 5):
        raise ValueError("max_values_matrix 必须是一个形状为 (5, 5, 5, 5) 的 NumPy 矩阵。")

    # 检查 indices_of_interest 是否是一个列表
    indices_of_interest = list(indices_of_interest)
    if not isinstance(indices_of_interest, list):
        raise ValueError("indices_of_interest 必须是一个索引列表。")

    # 创建空列表来存储关心的值
    values_of_interest = []

    # 设置图像的宽度和高度以实现16:9长宽比
    plt.figure(figsize=(16, 5))

    # 遍历索引列表，从 max_values_matrix 中提取值
    for index in indices_of_interest:
        if 0 <= index < 625:
            values_of_interest.append(max_values_matrix.flat[index])
        else:
            raise ValueError(f"索引 {index} 超出了 max_values_matrix 的范围。")

    if plot == False:
        # 绘制散点图，横坐标使用 indices_of_interest
        plt.scatter(indices_of_interest, values_of_interest, label='Values of Interest', color='blue')
    else:
        plt.plot(indices_of_interest, values_of_interest, label='Values of Interest', color='blue',marker="o")

    plt.xlabel('Indices of Interest')
    plt.ylabel('Max Value')
    plt.title(f'{title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像为PNG文件并显示
    input_filename = os.path.join(output_dir, f"{title}.png")
    plt.savefig(input_filename)
    plt.show()
    plt.close()  # 关闭图形以便下次绘制


def compare_best_valid_acc(input_dir1="csv文件所在位置",input_dir2="第二个csv文件所在的位置",  output_dir="png文件输出的位置", plot=False, use_text=True):

    def get_best_acc_from_csv(input_dir):
        # 创建字典用于保存最大值
        max_values = {}

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        # 列出包含数据的CSV文件
        path_list = os.listdir(input_dir)
        path_list.sort(key=lambda x:int(x.split('-')[0]) )
        csv_files = [file for file in path_list if file.endswith('.csv')]
        
        # 循环处理每个CSV文件
        for i, csv_file in enumerate(csv_files):
            print(f"{csv_file}")
            scale_key = csv_file.split(".csv")[0]
            # 构建CSV文件的完整路径
            csv_path = os.path.join(input_dir, csv_file)

            # 读取CSV文件为DataFrame
            df = pd.read_csv(csv_path)

            # 获取横坐标和纵坐标数据
            # x_data = df.iloc[:, 4]  # 第二列作为横坐标
            x_data = range(len(df.iloc[:,4]))
            y_data = df.iloc[:, 4]  # 第三列作为纵坐标
            # 计算最大值并保存到字典
            max_value = y_data.max()
            # key = f"scale=0.{scale_index}"
            max_values[scale_key] = max_value

        # 打印保存的最大值字典
        print("最大值字典:")
        for key, value in max_values.items():
            print(f"{key}: {value}")

        return max_values

    difference_dict = {}
    max_values1 = get_best_acc_from_csv(input_dir1)
    max_values2 = get_best_acc_from_csv(input_dir2)
    # max_values = max_values1 - max_values2
    for key in max_values1:
        if key in max_values2:
            difference_dict[key] = max_values1[key] - max_values2[key]
    # 绘制子图
    if plot:
        plt.plot(range(len(difference_dict.keys())), difference_dict.values(), marker="*")
    else:
        plt.scatter(range(len(difference_dict.keys())), difference_dict.values() )

    if use_text:
        for i, j in zip(range(len(difference_dict.keys())), difference_dict.values()):
            plt.text(i, j, f'({j})', fontsize=8, ha='center', va='bottom')
    plt.title(f"{input_dir1.split('-')[0]}-{input_dir2.split('-')[0]}做差比较准确率")  # 使用文件名作为子图标题
    plt.ylabel("max_valid_accuracy")  #为横坐标轴标签
    plt.xlabel("scale")  # 使用第三列列名作为纵坐标轴标签
    plt.grid()

    # 保存子图
    input_filename = os.path.join(output_dir, "比较重方法的优劣.png")  # 文件名与题目一致，去除扩展名并添加.png后缀
    plt.savefig(input_filename)
    # print(f"Plot saved as '{input_filename}'")