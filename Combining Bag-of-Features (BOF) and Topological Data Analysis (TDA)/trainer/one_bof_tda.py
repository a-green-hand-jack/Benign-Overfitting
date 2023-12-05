# 这里的目的是得到一种情况下的所有的BOF和TDA

# %% # 首先加载合适的库
from typing import Dict, Union, List, Any, Tuple
import torch
import numpy as np
import os
import pickle
import pandas as pd
import torch.nn.functional as F
from pprint import pprint
import pickle
# 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
try:
    import ripserplusplus as rpp_py
except ImportError:
    from ripser import ripser

from TDA.get_dataloader import get_dataloader,loader2vec, vec_dis
from TDA.data2betti import distance_betti, distance_betti_ripser
from post_process.tools4draw import plt_betti_number,plot_stacked_horizontal_bars
from TDA.after_betti import calculate_edge_length, get_min_max_columns, count_epsilon_bar_number,get_max_death
from ripser import Rips, ripser
from nets.simple_net import MLP,LeNet,ModelInitializer, init_weights, mixup_data
from BOF.get_rank_from_matrix import Effective_Ranks
from trainer.train_net import get_best_test_acc
# from post_process.betti_feature_compare import try_load_pkl


# %% ## 得到一次增强下的某一个model的输出

# 首先是一个函数，这个函数吸收一个model和一种增强的一种强度，然后会根据条件返回input+output或者input+hidden layers +output。为了减小计算上和子集造成的不确定性，需要反复计算10遍得到平均值，然后返回。



class ModelWithOneAugmentation:
    def __init__(self,
             model: Any = None,
             save_root: str = "./distance/Net-test/",
             chose: str = "cifar10_debug",
             debug_size: int = 1000,
             device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
             net_name: str = "Net",
             augmentation_name: str = "Angle",
             transform: Any = None,
             alpha: float = 0.0,
             gpu_flag: bool = True,
             num_repeats: int = 10,
             inner_layer: bool = False,
             save_path: str = "",
             num_epochs: int = 5,
             train_model: bool = False) -> None:
        """
            初始化函数，用于设置类的初始属性。

            Args:
            - model (Any, optional): 模型，默认为None
            - save_root (str, optional): 保存根目录，默认为"./distance/Net-test/"
            - chose (str, optional): 选择项，默认为"cifar10_debug"
            - debug_size (int, optional): 调试大小，默认为1000
            - device (torch.device, optional): 设备，默认为torch.device("cuda" if torch.cuda.is_available() else "cpu")
            - net_name (str, optional): 网络名称，默认为"Net"
            - augmentation_name (str, optional): 增强名称，默认为"Angle"
            - transform (Any, optional): 变换，默认为None
            - alpha (float, optional): Alpha值，默认为0.0
            - gpu_flag (bool, optional): GPU标志，默认为True
            - num_repeats (int, optional): 重复次数，默认为10
            - inner_layer (bool, optional): 是否使用内部层，默认为False
            - save_path (str, optional): 保存路径，默认为空字符串，应为路径格式

            Returns:
            - None
            """
        # 输入的部分
        self.model = model
        self.save_root = save_root
        self.debug_size = debug_size
        self.device = device
        self.net_name = net_name
        self.augmentation_name = augmentation_name
        self.transform = transform
        self.train_loader = None
        self.test_loader = None

        # 输出的部分
        self.averaged_tensors_list = []     # 所有特征图的平均值
        self.all_l2_distances = []  # 所有特征图的L2范式下的距离，在算距离之前对每一个样本做了归一化处理
        self.all_l1_distances = []  # 所有特征图的L1范式下的距离，在算距离之前对每一个样本做了归一化处理
        self.betti_features = {}
        
        # 操作的部分
        self.get_layer_output()
        self.get_data_distance()
        # print(len(self.all_l2_distances))
        self.l2_betti = self.form_distance_to_betti(self.all_l2_distances)  # 得到了L2范式定义距离下的betti number ,而且是不同层的
        self.l1_betti = self.form_distance_to_betti(self.all_l1_distances)  # 得到了L1范式定义距离下的betti number， 而且是不同层的
        self.betti_features[f"{net_name}+{augmentation_name}+L1"] = self.get_betti_features(layer_betti_number_list=self.l1_betti)
        self.betti_features[f"{net_name}+{augmentation_name}+L2"] = self.get_betti_features(layer_betti_number_list=self.l2_betti)

        self.BOF = self.get_BOF()
        if train_model:
        # 得到当前model 和增强下的在test dataset上的最佳预测准确度，主要要保证增强被正确的使用
            self.train_loader,self.test_loader = get_dataloader(chose="cifar10",transform=self.transform)
            self.best_test_acc = {f"{net_name}+{augmentation_name}+best_test_acc":get_best_test_acc(train_loader=self.train_loader, test_loader=self.test_loader, net=self.model, num_epochs=num_epochs,patience=50)}
        else:
            self.best_test_acc = {f"{net_name}+{augmentation_name}+best_test_acc":0.0}
            

        # 保存当前得到的特征
        self.save_all_features(save_path=save_path)
        # 绘制betti bars
        self.draw_betti_bars(save_root=save_path, distance_type="L1",layer_betti_bar=self.l1_betti)
        self.draw_betti_bars(save_root=save_path, distance_type="L2",layer_betti_bar=self.l2_betti)
        

    def get_layer_output(self,
                    chose: str = "cifar10_debug",
                    alpha: float = 0.0,
                    num_repeats: int = 10) -> None:
        """Obtains output from each layer of the model and calculates their averaged tensors.

        Args:
            chose (str): The dataset choice. Defaults to 'cifar10_debug'.
            alpha (float): Mixup parameter. Defaults to 0.0.
            num_repeats (int): Number of repeats. Defaults to 10.

        Returns:
            List[Tensor]: A list containing averaged tensors for each layer.

        Raises:
            SomeError: Description of error condition.

        Note:
            Additional notes about the function.
        """

        model_name = type(self.model).__name__
        print(f"This is {model_name}!!!")
        self.model.to(self.device)
        self.model.apply(init_weights)
        self.model.eval()

        train_loader, test_loader = get_dataloader(chose=chose, debug_size=self.debug_size, transform=self.transform)

        
        # self.test_loader = test_loader
        
        all_layers_output = []

        for repeat in range(num_repeats):
            # 遍历数据集并存储每一层的输出
            with torch.no_grad():
                # 初始化一个列表，每个元素都是一个空的张量，用于存储每一层的输出
                layer_outputs = [torch.tensor([]) for _ in range(1 + len(list(self.model.children())))]
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Check whether to perform operations on GPU
                    if alpha > 0.0 and alpha <= 0.5:
                        mixed_data = mixup_data(data, alpha)
                    else:
                        mixed_data = data
                    mixed_data, target = mixed_data.to(self.device), target.to(self.device)
                    outputs = self.model(mixed_data)
                    # 将每一层的输出追加到对应的张量中
                    for i, layer_output in enumerate(outputs):
                        # 将当前批次的输出连接到之前存储的张量中
                        layer_outputs[i] = torch.cat((layer_outputs[i], layer_output.cpu()), dim=0)
            # 现在 layer_outputs 中每个张量都包含了对应层所有批次输出的连接
            all_layers_output.append(layer_outputs)

        # 两层的list转化为df数据以便于操作
        all_layers_output_df = pd.DataFrame(all_layers_output)
        # 用于存储每列张量的平均值

        # 遍历每一列，对每列的张量进行平均值计算
        for column in all_layers_output_df.columns:
            # 将 DataFrame 中的张量转换为 PyTorch 张量
            tensors_in_column = all_layers_output_df[column].values.tolist()
            # 计算每列张量的平均值
            averaged_tensor = torch.stack(tensors_in_column).mean(dim=0)
            self.averaged_tensors_list.append(averaged_tensor)
         
    def get_data_distance(self) -> None:
        """Calculates L1 and L2 distances for averaged tensors.

        This function computes L1 and L2 distances for the averaged tensors obtained previously, 
        which contain input, output, and inner layers' feature maps. The distances are computed 
        as per the L1 and L2 norms and are stored in separate lists.

        Args:
            None

        Returns:
            None

        Raises:
            Any exceptions raised during the distance calculation.

        Note:
            The computed distances are stored in self.all_l1_distances and self.all_l2_distances lists.
        """
        for layer_number, layer_output in enumerate(self.averaged_tensors_list):
            concatenated_outputs = layer_output.view(layer_output.shape[0], -1)

            l2_distances = vec_dis(data_matrix=concatenated_outputs, distance="l2", root=self.save_root)

            l1_distances = vec_dis(data_matrix=concatenated_outputs, distance="l1", root=self.save_root)
            
            self.all_l2_distances.append(l2_distances)
            self.all_l1_distances.append(l1_distances)

    def form_distance_to_betti(self, distance_matrix_list: List[Any]) -> List[Union[Any, List[Any]]]:
        """Calculates Betti numbers from a list of distance matrices.

        This function computes the Betti numbers corresponding to each distance matrix
        provided in the input list. It returns a list where each element corresponds to
        the Betti numbers of a specific distance matrix. The output can be controlled to
        include Betti numbers of specific dimensions; currently set to include 0th and 1st order.

        Args:
            distance_matrix_list (List[Any]): A list containing distance matrices.

        Returns:
            List[Union[Any, List[Any]]]: A list where each element represents the Betti numbers
            of a particular distance matrix.

        Raises:
            Any exceptions that might occur during the calculation process.

        Note:
            The function checks for the existence of 'rpp_py' in globals() to determine whether
            to use 'rpp_py' or 'ripser' for the computation.
        """
        betti_number_list = []  # 吸收每一个特征图的的距离矩阵的betti numer
        for l_distance_matrix in distance_matrix_list:
            # 判断是否安装了 ripser++，如果安装了就使用 ripserplusplus，否则使用 ripser
            if 'rpp_py' in globals():
                d1 = rpp_py("--format distance --dim 1", l_distance_matrix)
            else:

                d1 = ripser(l_distance_matrix, maxdim=1, distance_matrix=True)
                d1 = d1["dgms"]
            d1 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix) for matrix in d1]  # 使用推导式，betti number中的那些无限大的采用最大值代替
            normalized_d1 = [np.where(np.isinf(matrix), np.nanmax(matrix[np.isfinite(matrix)]), matrix / np.nanmax(matrix[np.isfinite(matrix)])) for matrix in d1]

            betti_number_list.append(normalized_d1)
        return betti_number_list
        
    def get_betti_features(self, layer_betti_number_list: List[List[np.ndarray]]) -> Dict[str, List[Dict[int, Union[int, float]]]]:
        """
        获取Betti特征信息。

        Args:
        - layer_betti_number_list (List[List[np.ndarray]]): 两层嵌套的Betti Number列表。

        Returns:
        - betti_features (Dict[str, List[Dict[int, Union[int, float]]]]): 包含不同特征的字典。
        """
        betti_features: Dict[str, List[Dict[int, Union[int, tuple]]]] = {
            "bar_number": [],
            "all_bars_survive_time_sum": [],
            "max_epsilon_bar_number": [],
            "death_len": [],
            "max_death":[]
        }

        for layer_number, betti_number_list in enumerate(layer_betti_number_list):
            for index_betti, betti_number_matrix in enumerate(betti_number_list):
                bar_number = betti_number_matrix.shape[0]
                all_bars_survive_time_sum = np.sum(betti_number_matrix[:, 1] - betti_number_matrix[:, 0])
                max_epsilon_bar_number = count_epsilon_bar_number(betti_number_matrix)
                birth_len, death_len = calculate_edge_length(betti_number_matrix)
                max_death = get_max_death(betti_number_matrix)

                betti_features["bar_number"].append({f"{index_betti}th_dim": (layer_number, bar_number)})
                betti_features["all_bars_survive_time_sum"].append({f"{index_betti}th_dim": (layer_number,all_bars_survive_time_sum)})
                betti_features["max_epsilon_bar_number"].append({f"{index_betti}th_dim": (layer_number,max_epsilon_bar_number)})
                betti_features["death_len"].append({f"{index_betti}th_dim": (layer_number,death_len)})
                betti_features["max_death"].append({f"{index_betti}th_dim": (layer_number,death_len)})

        return betti_features

    def get_BOF(self):
        
        # print(type(self.averaged_tensors_list[0]), "\n--------\n", self.averaged_tensors_list[0].shape)
        reshaped_tensor = self.averaged_tensors_list[0].reshape(self.averaged_tensors_list[0].shape[0], -1)

        get_rank = Effective_Ranks(reshaped_tensor.numpy())

        r0 = get_rank.r0
        R0 = get_rank.R0
        rk_max_index = get_rank.rk_max_index
        rk_max = get_rank.rk_max_value
        Rk_max = get_rank.Rk_value_max_rk_index

        return {f"{self.net_name}+{self.augmentation_name}":{"r0":r0, "R0":R0, "rk_max_index":rk_max_index, "rk_max":rk_max, "Rk_max":Rk_max}}

    def save_all_features(self, save_path: str) -> Tuple[str, str, str]:
            """
            将重要信息保存在本地路径中以备操作。该函数将数据分别保存在独立的.pkl文件中。

            Args:
            - save_path (str): 要保存文件的路径

            Returns:
            - Tuple[str, str, str]: 保存的三个文件的路径

            Raises:
            - OSError: 如果保存路径无效

            注：假设 self.betti_features, self.best_test_acc, self.BOF 是要保存的数据
            """
            data_to_save = {
                'betti_features': self.betti_features,
                'best_test_acc': self.best_test_acc,
                'BOF': self.BOF
            }

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            betti_features_path = os.path.join(save_path, 'betti_features.pkl')
            best_test_acc_path = os.path.join(save_path, 'best_test_acc.pkl')
            BOF_path = os.path.join(save_path, 'BOF.pkl')

            with open(betti_features_path, 'wb') as f:
                pickle.dump(data_to_save['betti_features'], f)

            with open(best_test_acc_path, 'wb') as f:
                pickle.dump(data_to_save['best_test_acc'], f)

            with open(BOF_path, 'wb') as f:
                pickle.dump(data_to_save['BOF'], f)

            print(f"3种特征被保存在{betti_features_path}，{best_test_acc_path}，{BOF_path}中！！")


            return betti_features_path, best_test_acc_path, BOF_path

    def draw_betti_bars(self, save_root, layer_betti_bar, distance_type):
        # 考察的self.l2_betti是一个list，这个list保存着每一层的betti bars的数据结构如下： List[List[np.ndarray]]。List[np.ndarray]代表某一层的第0阶和第1阶的betti bars，也就是里面有两个np.ndarray，这个是一个N*2的矩阵
        # 保存的时候，命名是f"{L_distance}{np_index}.png"

        # save_path =  f"{save_root}/{layer_number}/"

        for layer_index, layer_list in enumerate(layer_betti_bar):
            # 得到某一层
            # for dim_betti_index, dim_betti_matrix in enumerate(layer_list)
            png_title = f"{distance_type}_{self.augmentation_name}下{self.net_name}的{layer_index}层的betti_bars"
            png_save_path = f"{save_root}/{layer_index}th_layer_betti_{distance_type}"

            plt_betti_number(betti_number=layer_list, plt_title=png_title, root=png_save_path)

            plot_stacked_horizontal_bars(betti_number=layer_list, plt_title=png_title, root=png_save_path)



    





