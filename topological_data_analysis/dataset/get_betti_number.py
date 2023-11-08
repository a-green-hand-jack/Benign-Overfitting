import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from dataset.get_dataloader import get_dataloader,loader2vec, vec_dis
from dataset.data2betti import distance_betti, distance_betti_ripser, plt_betti_number,plot_betti_number_bars
from ripser import Rips, ripser

def betti_number(chose="cifar10_debug"):
    train_loader, test_loader = get_dataloader(chose=chose)
    flattened_images = loader2vec(train_loader=train_loader)

    # flattened_images现在包含整个训练集中的图像向量，形状为(N, 3 * 224 * 224)，其中N是训练集的大小
    l2_distances = vec_dis(data_matrix=flattened_images, distance="l2_db",save_flag=True)
    l1_distances = vec_dis(data_matrix=flattened_images, distance="l1_db",save_flag=True)
    # print(l2_distances)


    # 读取 L2 范数距离矩阵
    loaded_l2_distances = np.load('distance\\l2_db_distance.npy')

    # 读取 L1 范数距离矩阵
    loaded_l1_distances = np.load('distance\\l1_db_distance.npy')

    d1= ripser(loaded_l1_distances, maxdim=1, distance_matrix=True)
    plt_betti_number(d1["dgms"],plt_title="L1")
    # print(len(d["dgms"][0]))

    plot_betti_number_bars(d1["dgms"],plt_title="L1")

    d2= ripser(loaded_l2_distances, maxdim=1, distance_matrix=True)
    plt_betti_number(d2["dgms"],plt_title="L2")
    # print(len(d["dgms"][0]))

    plot_betti_number_bars(d2["dgms"],plt_title="L2")


if __name__ == '__main__':

    betti_number()