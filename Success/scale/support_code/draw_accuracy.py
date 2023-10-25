import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

def draw_accuracy(input_dir="csv文件所在位置", output_dir="png文件输出的位置"):
    '''
    输入通过tensorboard处下载的csv文件所在的文件夹,在同样的位置输出一张包含所有信息的大图
    '''

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
        plt.plot(x_data, y_data )
        plt.title(f"{csv_file}")  # 使用文件名作为子图标题
        plt.xlabel(df.columns[1])  # 使用第二列列名作为横坐标轴标签
        plt.ylabel(df.columns[2])  # 使用第三列列名作为纵坐标轴标签
        plt.grid()

        # 保存子图
        input_filename = os.path.join(output_dir, f"{csv_file.split('.csv')[0]}.png")  # 文件名与题目一致，去除扩展名并添加.png后缀
        plt.savefig(input_filename)
        # print(f"Plot saved as '{input_filename}'")


def best_valid_acc(input_dir="csv文件所在位置",  output_dir="png文件输出的位置"):
    # 列出包含数据的CSV文件
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # 创建字典用于保存最大值
    max_values = {}

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # 列出包含数据的CSV文件
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # 循环处理每个CSV文件
    for i, csv_file in enumerate(csv_files):
        scale_key = float(csv_file.split("val")[1].split(".csv")[0])
        # 构建CSV文件的完整路径
        csv_path = os.path.join(input_dir, csv_file)

        # 读取CSV文件为DataFrame
        df = pd.read_csv(csv_path)

        # 获取横坐标和纵坐标数据
        x_data = df.iloc[:, 1]  # 第二列作为横坐标
        y_data = df.iloc[:, 2]  # 第三列作为纵坐标
         # 计算最大值并保存到字典
        max_value = y_data.max()
        # key = f"scale=0.{scale_index}"
        max_values[scale_key] = max_value

    # 打印保存的最大值字典
    print("最大值字典:")
    for key, value in max_values.items():
        print(f"{key}: {value}")

    # 绘制子图
    plt.plot(max_values.keys(), max_values.values() )
    plt.title("观察valid中最大准确率的变化")  # 使用文件名作为子图标题
    plt.ylabel("max_valid_accuracy")  #为横坐标轴标签
    plt.xlabel("scale")  # 使用第三列列名作为纵坐标轴标签
    plt.grid()

    # 保存子图
    input_filename = os.path.join(output_dir, "观察valid中最大准确率的变化.png")  # 文件名与题目一致，去除扩展名并添加.png后缀
    plt.savefig(input_filename)
    # print(f"Plot saved as '{input_filename}'")