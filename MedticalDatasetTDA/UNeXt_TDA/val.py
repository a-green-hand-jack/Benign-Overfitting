import os
from glob import glob
import statistics
import random

import cv2
from pprint import pprint
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.linear import Linear
import pickle
# 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
try:
    import ripserplusplus as rpp_py
except ImportError:
    from ripser import ripser

from UNeXt_TDA.dataset_copy import Dataset, JointTransform2D
from UNeXt_TDA.utils import AverageMeter
from UNeXt_TDA.archs import UNext
from TDA.get_dataloader import vec_dis
from TDA.after_betti import calculate_edge_length
from dataset.data2betti import distance_betti, distance_betti_ripser, plt_betti_number,plot_betti_number_bars


def init_weights(m: nn.Module) -> None:
    """
    初始化神经网络模型的权重和偏置（如果存在）。

    Args:
    - m (nn.Module): 需要初始化权重和偏置的神经网络模型

    Returns:
    - None
    """
    # 设置随机数种子
    seed = 0
    torch.manual_seed(seed)  # 设置torch的随机数种子
    random.seed(seed)  # 设置python的随机数种子
    np.random.seed(seed)  # 设置numpy的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置cuda的随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有cuda设备的随机数种子

    if isinstance(m, (_ConvNd, Linear)):  # 检查是否是卷积层或线性层
        init.normal_(m.weight.data, mean=0, std=0.01)  # 初始化权重为均值为0，标准差为0.01的正态分布
        if isinstance(m, _ConvNd) and m.bias is not None:
            init.constant_(m.bias.data, 0)  # 如果是卷积层且有偏置项，初始化偏置为常数0
        elif isinstance(m, Linear) and hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0)  # 如果是线性层且有偏置项，初始化偏置为常数0


    
class UNeXtTDA:
    def __init__(self, 
        save_file_path=None, 
        repetitions=1, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        betti_dim = 1,
        model_name = 'UNeXt',
        input_images_path = '..\\..\\others_work\\dataset\\ISIC2018\\train_folder\\images',
        input_masks_path = '..\\..\\others_work\\dataset\\ISIC2018\\train_folder\\masks',
        # output_path = os.path.join('outputs', 'isic_crop_512'),
        crop = 512,
        img_ext = '.png',
        mask_ext = '.png'):
        # init_weights()
        # 设置随机数种子
        seed = 0
        torch.manual_seed(seed)  # 设置torch的随机数种子
        random.seed(seed)  # 设置python的随机数种子
        np.random.seed(seed)  # 设置numpy的随机数种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # 设置cuda的随机数种子
            torch.cuda.manual_seed_all(seed)  # 设置所有cuda设备的随机数种子

        # 加载需要的各种变量
        self.repetitions = repetitions
        self.device = device
        self.images_matrix = None  # 初始化为 None
        self.model_name = model_name
        self.input_images_path = input_images_path
        self.input_masks_path = input_masks_path
        # self.output_path = output_path
        self.crop = crop
        self.img_ext = img_ext
        self.mask_ext = mask_ext

        # 得到了两种距离下的betti number list
        L_12_betti_bars = self.get_betti_number(save_root = save_file_path)
        # 保存0th的betti number的特征情况
        betti_dim = 0
        self.feature2save = self.betti_number_feature(chose='L1', betti_dim=betti_dim)
        self.save_stats_to_file(file_path=save_file_path, betti_dim=betti_dim)  
        self.feature2save = self.betti_number_feature(chose='L2', betti_dim=betti_dim)
        self.save_stats_to_file(file_path=save_file_path ,chose="L2", betti_dim=betti_dim)

        # 保存1th的betti number的特征情况
        betti_dim = 1
        self.feature2save = self.betti_number_feature(chose='L1', betti_dim=betti_dim)
        self.save_stats_to_file(file_path=save_file_path, betti_dim=betti_dim)
        self.feature2save = self.betti_number_feature(chose='L2', betti_dim=betti_dim)
        self.save_stats_to_file(file_path=save_file_path ,chose="L2", betti_dim=betti_dim)

        # 保存betti bars 的数据，以便后期进行其他处理
        self.feature2save = L_12_betti_bars
        self.save_stats_to_file(file_path=save_file_path ,chose="bars", betti_dim=12)


    # def images_to_matrix_lists(self):
    def images_to_matrix_lists(self):

        cudnn.benchmark = True
        print("=> creating model".format(self.model_name))
        model = UNext(num_classes=1, input_channels=3, deep_supervision='False')
        # 对模型的每个参数应用随机初始化函数
        model.apply(init_weights)
        model = model.cuda()
        # model.load_state_dict(torch.load(pre_trained_model_path))
        model.eval()

        # Data loading code
        
        img_ids =  glob(os.path.join(self.input_images_path, '*' + self.img_ext))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
        tf_val = JointTransform2D(crop=(self.crop, self.crop), p_flip=0, color_jitter_params=None, long_mask=True)
        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=self.input_images_path,
            mask_dir=self.input_masks_path,
            img_ext=self.img_ext,
            mask_ext=self.mask_ext,
            num_classes=1,
            transform=tf_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False)

        images_matrix_lists = []
        with torch.no_grad():
            for _ in range(self.repetitions):
                torch.cuda.empty_cache()
                image_vectors = []  # Storing image vectors for each iteration
                for input, target, meta in val_loader:
                    input = input.cuda()
                    model = model.cuda()
                    # compute output
                    # image = model(input)
                    image = input
                    # image = torch.sigmoid(image).cpu().numpy()
                    image_vector = image.flatten()  # 转换为向量
                    image_vector_array = image_vector.cpu().detach().numpy()
                    image_vectors.append(image_vector_array)
                
                image_matrix = np.vstack(image_vectors)  # Stack image vectors to form a matrix
                images_matrix_lists.append(image_matrix)  # Append the matrix to the list
        return images_matrix_lists
    
    def img_matrix2distance_matrix(self):
       
        images_matrix_list = self.images_to_matrix_lists()  # 转换图像到矩阵列表
        self.all_l2_distances = []
        self.all_l1_distances = []

        for imgmatrix in images_matrix_list:
            l2_distances = vec_dis(data_matrix=torch.from_numpy(imgmatrix).to(self.device), distance="l2")
            l1_distances = vec_dis(data_matrix=torch.from_numpy(imgmatrix).to(self.device), distance="l1")
            self.all_l2_distances.append(l2_distances)
            self.all_l1_distances.append(l1_distances)

        return self.all_l1_distances, self.all_l2_distances


    def get_betti_number(self,save_root):
           
        self.l1_betti_number_list = []  # 吸收每一个特征图的的距离矩阵的betti numer
        self.l2_betti_number_list = []
        for l1_distance_matrix, l2_distance_matrix in zip(*self.img_matrix2distance_matrix()):    # 运行函数得到了L1，L2距离矩阵list
            # 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
            if 'rpp_py' in globals():
                d1 = rpp_py.run("--format distance --dim 1", l1_distance_matrix.cpu().detach().numpy())
                d2 = rpp_py.run("--format distance --dim 1", l2_distance_matrix.cpu().detach().numpy())
                # 假设 d1 和 d2 是包含元组的字典
                # 转化为3维矩阵的过程
                d1_matrix = []
                for key in d1:
                    d1_matrix.append(np.array([list(item) for item in d1[key]]))
                d2_matrix = []
                for key in d1:
                    d2_matrix.append(np.array([list(item) for item in d1[key]]))
                d1 = d1_matrix
                d2 = d2_matrix
            else:

                d1 = ripser(l1_distance_matrix, maxdim=1, distance_matrix=True)
                d1 = d1["dgms"]
                d2 = ripser(l2_distance_matrix, maxdim=1, distance_matrix=True)
                d2 = d2["dgms"]
            
            # 对两种定义下的距离矩阵进行后处理
            d1 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix) for matrix in d1]  # 使用推导式，betti number中的那些无限大的采用最大值代替
            d1 = [np.nan_to_num(matrix, nan=0.0) for matrix in d1]  # 将NaN值替换为0
            d1 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix) if matrix.size > 0 else matrix for matrix in d1]
            normalized_d1 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix / np.nanmax(matrix[np.isfinite(matrix)])) for matrix in d1]  # 实现betti number 层面上的归一化
            self.l1_betti_number_list.append(normalized_d1)

            d2 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix) for matrix in d2]  # 使用推导式，betti number中的那些无限大的采用最大值代替
            d2 = [np.nan_to_num(matrix, nan=0.0) for matrix in d2]  # 将NaN值替换为0
            normalized_d2 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix / np.nanmax(matrix[np.isfinite(matrix)])) for matrix in d2]  # 实现betti number 层面上的归一化            
            self.l2_betti_number_list.append(normalized_d2)

        # plt_betti_number(d1, plt_title="L1", root=save_root)
        # plot_betti_number_bars([d1, d2], plt_title="L1", root=save_root)
        # plt_betti_number(d2, plt_title="L2", root=save_root)
        # plot_betti_number_bars(d2, plt_title="L2", root=save_root)


        return {"L1_betti_number_list":self.l1_betti_number_list, "L2_betti_number_list":self.l2_betti_number_list}    # 这个list中的每一个元素，矩阵，都代表了一个数据集

    def betti_number_feature(self, chose="L1", betti_dim = 1):
        if chose == "L1":
            betti_number_list = self.l1_betti_number_list
        elif chose == "L2":
            betti_number_list = self.l2_betti_number_list
        all_bars_survive_time_sum_list = []
        death_len_list = []
        average_array_list = []
        if betti_dim == 1:
            for betti_number_matrix in betti_number_list:
                
                betti_number_matrix = betti_number_matrix[1]
                # print(betti_number_matrix)
                # 现在只关心all_bars_survive_time_sum和death_len
                all_bars_survive_time_sum = np.sum(betti_number_matrix[:, 1] - betti_number_matrix[:, 0])
                birth_len, death_len = calculate_edge_length(betti_number_matrix)

                all_bars_survive_time_sum_list.append(all_bars_survive_time_sum)
                death_len_list.append(death_len)
                average_array_list.append(betti_number_matrix)
                # print(betti_number_matrix.shape, "\n")
            # print(len(average_array_list))
            # average_arry = np.mean(np.array(average_array_list), axis=0)
            # average_arry = average_array_list[0]
        else:
            for betti_number_matrix in betti_number_list:
                
                betti_number_matrix = betti_number_matrix[0]    # 计算的是0阶bitt number的表现
                # print(betti_number_matrix)
                # 现在只关心all_bars_survive_time_sum和death_len
                all_bars_survive_time_sum = np.sum(betti_number_matrix[:, 1] - betti_number_matrix[:, 0])
                birth_len, death_len = calculate_edge_length(betti_number_matrix)

                all_bars_survive_time_sum_list.append(all_bars_survive_time_sum)
                death_len_list.append(death_len)
                average_array_list.append(betti_number_matrix)
            #     print(betti_number_matrix.shape, "\n")
            # print(len(average_array_list))
            # print(average_array_list)
            # average_arry = np.mean(np.array(average_array_list), axis=0)
            

        results = {}
        
        if self.repetitions == 1:
            results["death_len"] = (statistics.mean(death_len_list),0.0)
            results["all_bars_survive_time_sum"] = (statistics.mean(all_bars_survive_time_sum_list), 0.0)  # 存储均值和标准差为元组形式
        else:
            results["death_len"] = (statistics.mean(death_len_list), statistics.stdev(death_len_list))
            results["all_bars_survive_time_sum"] = (statistics.mean(all_bars_survive_time_sum_list), statistics.stdev(all_bars_survive_time_sum_list))  # 存储均值和标准差为元组形式
        # results["average_betti_number_matrix"] = average_arry
        return results

    
    def save_stats_to_file(self, file_path, chose="L1", betti_dim = 1):
        # self.BOF_mean_stddev = self.calculate_stats()
        betti_features_path = os.path.join(file_path, f'{chose}_betti_features_{betti_dim}th.pkl')
        # folder_path = os.path.dirname(file_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        with open(betti_features_path, 'wb') as f:
                pickle.dump(self.feature2save, f)

if __name__ == '__main__':
    # pass
    temp_img = UNeXtTDA(repetitions=10, save_file_path = '.\\test', betti_dim=1)
    print(temp_img.feature2save)

