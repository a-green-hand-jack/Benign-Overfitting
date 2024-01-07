# !pip install ripserplusplus
# !pip install ripser

# 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
try:
    import ripserplusplus as rpp_py
except ImportError:
    from ripser import ripser
    
import numpy as np
from tqdm import tqdm
import sys
from ripser import ripser
import time
import matplotlib.pyplot as plt
import os


def distance_betti(distances=None):
    start = time.time()
    num_iters= len(distances)

    d = rpp_py.run("--format distance", distances)

    end = time.time()
    # print("ripser++ total time: ", end-start)

    return d


def distance_betti_ripser(distances=None):
    start = time.time()

    d= ripser(distances, maxdim=2, distance_matrix=True)
    # print(d)
    end = time.time()
    # print("ripser.py total time", end-start)
    return d


def plt_betti_number(betti_number,plot=False,plt_title=None,root=None):
    ''' 
    Function: plt_betti_number

    Description: This function plots the Betti numbers using matplotlib. It can either plot the points as a line graph or as scattered points depending on the input parameters.

    Parameters: - betti_number: A list of numpy arrays where each array contains the x and y coordinates of the Betti numbers. - plot (optional): A boolean value indicating whether to plot the points as a line graph (default is False). - plt_title (optional): A string representing the title of the plot. - root (optional): A string representing the root directory where the plot image will be saved.

    Returns: None

    Save Path: The plot image will be saved at the specified root directory with the file name "{plt_title}_scatter.png".

    Example Usage: betti_number = [array([[0.5, 1.5], [1, 2.5], [1.5, 2]])] plt_betti_number(betti_number, plot=True, plt_title="Betti Numbers") 
    '''
    # 创建一个新图表
    if not os.path.exists(root):
            # 如果文件夹不存在，则创建它
            os.makedirs(root)
    plt.figure()
    save_path = f"{root}_{plt_title}_scatter.png"

    if plot:
        for index, value in enumerate(betti_number):
            x = value[:, 0]  # 提取横坐标
            y = value[:, 1]  # 提取纵坐标
            plt.plot(x, y, marker='o', label=f'H_{index}')
    else:
        for index, value in enumerate(betti_number):
            x = value[:, 0]  # 提取横坐标
            y = value[:, 1]  # 提取纵坐标
            plt.scatter(x, y, marker='o', label=f'H_{index}')

    # 添加图例
    plt.legend(loc="lower right")

    # 设置坐标轴标签
    plt.xlabel('Birth')
    plt.ylabel('Death')
    if plt_title is not None:
        plt.title(plt_title)

    plt.savefig(save_path)
    # 显示图表
    # plt.show()
    plt.close()


def plot_stacked_horizontal_bars(bar_data, index_title,plt_title,root=None):
    """
    绘制表示持续区间的堆叠水平条形图。

    参数:
    - bar_data (列表的列表): 表示持续区间的数据列表，每个元素是包含起始和结束位置的列表。
    - index_title (整数): 用于标题的索引。

    返回: 无返回值，但会显示绘制的水平条形图。
    """
    if not os.path.exists(root):
            # 如果文件夹不存在，则创建它
            os.makedirs(root)
    plt.figure()
    save_path = f"{root}_H{index_title}_{plt_title}_bar.png"

    # 初始化y坐标位置
    # y_positions = list(range(len(bar_data)))

    for index, bar in enumerate(bar_data):
        # print(bar)
        start = bar[0]  # 起始位置
        end = bar[1]    # 结束位置

        # 绘制水平条形图
        plt.barh(index, end - start, left=start, height=0.5)

    # 添加图例
    # plt.legend()

    # 设置坐标轴标签
    plt.ylabel('Number')
    plt.xlabel('Birth-Death')
    plt.title(f"H{index_title}_{plt_title}")
    plt.savefig(save_path)

    # 显示图表
    # plt.show()
    plt.close()

def plot_betti_number_bars(betti_number, bar_spacing=0.2,plt_title=None,root=None):
    """
    绘制表示 Betti 数的堆叠水平条形图。

    参数:
    - betti_number (列表的列表): 包含多个 Betti 数的列表，每个 Betti 数以堆叠水平条形图表示。
    - bar_spacing (浮点数): 条形图之间的间距。

    返回: 无返回值，但会显示绘制的堆叠水平条形图。
    """


    for index, value in enumerate(betti_number):
        # x = value[:, 0]  # 提取第一个数作为横坐标起始点
        # y = value[:, 1]  # 提取第二个数作为横坐标结束点

        plot_stacked_horizontal_bars(value, index,plt_title=plt_title,root=root)




