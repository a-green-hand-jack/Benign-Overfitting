import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']

def plot_csv_files_in_directory(directory, output_directory):
    # 创建一个空的 DataFrame 用于存储所有 CSV 文件的内容
    df = None
    first_csv = True  # 用于标记是否是第一个 CSV 文件

    # 获取目标文件夹中的所有 CSV 文件
    csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
    print(len(csv_files))

    # 创建输出目录
    os.makedirs(output_directory, exist_ok=True)

    # 遍历每个 CSV 文件并将其内容添加到 DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        
        # 读取 CSV 文件内容
        data = pd.read_csv(file_path, delimiter='\t')  # 假设 CSV 文件使用制表符分隔
        # print(data,'yyyyy')
        
        if first_csv:
            # 如果是第一个 CSV 文件，创建 DataFrame 并记录变量名
            df = data.copy()
            first_csv = False
        else:
            # 如果不是第一个 CSV 文件，只添加数据而不添加变量名
            df = pd.concat([df, data], ignore_index=True)

        # 绘制折线图（只有在 df 不为空时才绘制）
    
    if df is not None:
        print(df)
        # 绘制每一列的折线图
        for column_name in df.columns:
            # print(df[column_name][0])
            plt.figure()  # 创建一个新的图形
            plt.plot(df[column_name])  # 绘制列数据
            plt.xlabel('Index')  # 设置X轴标签
            plt.ylabel(column_name)  # 设置Y轴标签
            plt.title(f'{csv_file} - {column_name}')  # 设置图形标题
            plt.grid(True)  # 添加网格
            plt.savefig(os.path.join(output_directory, f'{csv_file}_{column_name}.png'))  # 保存图形为PNG文件

def scatter_plot_with_error_bars(data, x_labels=None, y_label=None, title=None, img_folder = 'img'):
    """
    绘制带有误差棒的散点图。

    参数:
    data (numpy.ndarray): 二维 numpy 矩阵，第一列为均值，第二列为方差。
    x_labels (list, optional): x 轴标签列表，与数据点一一对应。默认为 None。
    y_label (str, optional): y 轴标签。默认为 None。
    title (str, optional): 图表标题，默认为输入数据的变量名。

    返回:
    None
    """
    if data.shape[1] != 2:
        raise ValueError("输入数据必须是一个二维numpy矩阵，第一列为均值，第二列为方差。")

    x_values = range(len(data))
    y_values = data[:, 0]  # 提取均值列
    error_values = np.sqrt(data[:, 1])  # 提取方差列并计算标准差作为误差

    plt.errorbar(x_values, y_values, yerr=error_values, fmt='o', capsize=5, label='Data with Error Bars')

    if x_labels:
        plt.xticks(x_values, x_labels, rotation=45, ha='right')

    if y_label:
        plt.ylabel(y_label)

    if title is None:
        title = 'Scatter Plot with Error Bars'

    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图片到 "img" 文件夹
    
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img_filename = os.path.join(img_folder, f'{title}.png')
    plt.savefig(img_filename)

    plt.show()

def scatter_plot_mean(data,plot=False, x_labels=None, y_label=None, img_folder = 'img',x_data=None):
    """
    绘制带有误差棒的散点图。

    参数:
    data (numpy.ndarray): 二维 numpy 矩阵，第一列为均值，第二列为方差。
    x_labels (list, optional): x 轴标签列表，与数据点一一对应。默认为 None。
    y_label (str, optional): y 轴标签。默认为 None。
    title (str, optional): 图表标题，默认为输入数据的变量名。

    返回:
    None
    """
    data = data[x_data]
    if data.shape[1] != 2:
        raise ValueError("输入数据必须是一个二维numpy矩阵，第一列为均值，第二列为方差。")

    # x_values = range(len(data))
    y_values = data[:, 0]  # 提取均值列
    error_values = np.sqrt(data[:, 1])  # 提取方差列并计算标准差作为误差

    # plt.errorbar(x_values, y_values, yerr=error_values, fmt='o', capsize=5, label='Data with Error Bars')
    if plot == True:
        plt.plot(x_data, y_values, label='Mean data', marker='o')
    else:
        plt.scatter(x_data, y_values, label='Mean data')

    if x_labels:
        plt.xticks(x_data, x_labels, rotation=45, ha='right')

    if y_label:
        plt.ylabel(y_label)

    # if title is None:
    #     title = 'Scatter Plot with Error Bars'
    title = f"{x_data}"

    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图片到 "img" 文件夹
    
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img_filename = os.path.join(img_folder, f'{title}.png')
    plt.savefig(img_filename)

    plt.show()

def calculate_mean_and_variance(csv_folder):
    # 初始化一个DataFrame来保存所有CSV文件的数据
    all_data = pd.DataFrame()

    # 遍历文件夹中的所有CSV文件
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_folder, csv_file)
            try:
                # 读取CSV文件为DataFrame
                df = pd.read_csv(csv_path, delimiter=',')  # 这里的csv文件采用","实现分离
                
                # 计算每列的平均值和方差
                column_means = df.mean()
                column_variances = df.var()
                
                # 将结果添加到all_data中
                all_data = pd.concat([all_data, column_means.rename(f'{csv_file}_mean'), column_variances.rename(f'{csv_file}_variance')], axis=1)
            except Exception as e:
                print(f"Error processing file {csv_file}: {str(e)}")

    # 提取均值和方差的列名
    mean_columns = [column for column in all_data.columns if column.endswith("_mean")]
    variance_columns = [column for column in all_data.columns if column.endswith("_variance")]

    # 初始化一个列表来存储4个numpy矩阵
    matrix_list = []

    # 遍历每一行，将均值和方差数据分别存储到numpy数组中
    for _, row in all_data.iterrows():
        mean_values = row[mean_columns].values
        variance_values = row[variance_columns].values
        combined_data = np.column_stack((mean_values, variance_values))  # 将均值和方差列合并为一个numpy数组
        matrix_list.append(combined_data)

    # 将列表转换为numpy数组，每个元素对应一个矩阵
    result_matrices = np.array(matrix_list)
    
    return result_matrices



if __name__ == '__main__':
    # 调用函数并传递包含 CSV 文件的文件夹路径和输出目录路径
    input_directory_path = "your_input_directory_path_here"
    output_directory_path = "out_img"  # 输出目录名
    plot_csv_files_in_directory(input_directory_path, output_directory_path)
