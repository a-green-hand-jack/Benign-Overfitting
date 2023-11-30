# 这里是一些画图的工具
import numpy as np
from tqdm import tqdm
import sys
from ripser import ripser
import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示
import os
# 首先，是绘制betti bars
# 我发现自己似乎是忘记保存betti bars 本身了


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
    # if not os.path.exists(root):
    #         # 如果文件夹不存在，则创建它
    #         os.makedirs(root)
    plt.figure()
    save_path = f"{root}_scatter.png"

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


def plot_stacked_horizontal_bars(betti_number,plt_title,root=None):
    """
    绘制表示持续区间的堆叠水平条形图。

    参数:
    - bar_data (列表的列表): 表示持续区间的数据列表，每个元素是包含起始和结束位置的列表。
    - index_title (整数): 用于标题的索引。

    返回: 无返回值，但会显示绘制的水平条形图。
    """
    # if not os.path.exists(root):
    #         # 如果文件夹不存在，则创建它
    #         os.makedirs(root)
    plt.figure()

    save_path = f"{root}_bar.png"

    
    for index, bar_data in enumerate(betti_number):
        
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
    plt.title(f"{plt_title}")
    plt.savefig(save_path)

    # 显示图表
    # plt.show()
    plt.close()





