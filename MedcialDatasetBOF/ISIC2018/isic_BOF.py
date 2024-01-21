# 这里是为了得到isic2018对应的BOF的情况
# Add appropriate type hints for your transforms, Dataset, Effective_Ranks, and any other custom classes used in the code.

import os
import statistics
import pickle
from typing import List, Tuple, Dict, Any, Union
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体，中文显示
plt.rcParams['axes.unicode_minus'] = False   # 坐标轴负数的负号显示

from ISIC2018.isicdataset import Dataset, ImageToImage2D, JointTransform2D
from BOF.get_rank_from_matrix import Effective_Ranks, Effective_Ranks_GPU

class DynamicNormalize(transforms.Normalize):
    def __call__(self, tensor):
        # 动态计算平均值和方差
        mean = tensor.mean(dim=[1, 2], keepdim=True)
        std = tensor.std(dim=[1, 2], keepdim=True)
        
        # 使用动态计算得到的平均值和方差进行归一化
        return super().__call__(tensor, mean, std)



class ISICBOF:
    def __init__(
        self,
        save_file_path: str = None,
        repetitions: int = 2,
        costume_transform = [
                        transforms.RandomRotation(degrees=(0, 0))
                        ],
        imgpath: str = r'..\..\others_work\dataset\ISIC2018\train_folder'
    ) -> None:
        """
        Initialize the ISICBOF class.
        Args:
        - costume_transform: Transformation object for data augmentation.
        - save_file_path: Path to save the computed statistics.
        - repetitions: Number of iterations for image processing.
        - crop: Crop size for the images.
        - imgpath: Path to the image directory.
        """
        self.train_transform = costume_transform
        self.repetitions = repetitions
        self.imgpath = imgpath
        self.dataset = self.create_dataset()
        self.images_matrix_lists: List[np.ndarray] = []
        self.images_to_matrix_lists()  # Convert images to matrix lists
        self.BOF_feature_list = self.get_BOF()
        self.BOF_mean_stddev = self.calculate_stats_tensor()
        self.save_stats_to_file(file_path=save_file_path)

    def create_dataset(self) -> Any:
        """
        Create the dataset for image processing.
        Returns:
        - Dataset object.
        """
        img_ids = glob(os.path.join(self.imgpath, 'images', '*' + '.png'))
        print(self.imgpath)
        # print(img_ids)
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
        # tf_train = JointTransform2D(crop=(crop,crop), p_flip=0, color_jitter_params=None, long_mask=True)
        # 定义增强方法
        # augmentations = [
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        #     transforms.RandomAffine(degrees=180),
        # ]
        
        tf_train = JointTransform2D(augmentations=self.train_transform)

        train_dataset = Dataset(
            img_ids=train_img_ids,
            img_dir=os.path.join(self.imgpath, 'images'),
            mask_dir=os.path.join(self.imgpath, 'masks'),
            img_ext='.png',
            mask_ext='.png',
            num_classes=1,
            transform=tf_train
        )
        return train_dataset
    
    def images2matrix(self, trainset: Any) -> np.ndarray:
        """
        Convert images in the dataset to matrices.
        Args:
        - trainset: Dataset object.
        Returns:
        - Image matrix as numpy array.
        """
        image_vectors = []  # Storing image vectors for each iteration
        for i in range(len(trainset)):
            image, _, _ = trainset[i]  
            image_numpy = image.detach().cpu().numpy()  # 将PyTorch张量转换为NumPy数组
            image_vector = image_numpy.flatten()  # 转换为float32
            image_vectors.append(image_vector)
        image_matrix = np.vstack(image_vectors)
        return image_matrix

    def images2matrix_gpu(self, trainset: Any) -> torch.Tensor:
        """
        Convert images in the dataset to matrices.
        Args:
        - trainset: Dataset object.
        Returns:
        - Image matrix as PyTorch tensor on GPU if available, else on CPU.
        """
        image_vectors = []  # Storing image vectors for each iteration
        for i in range(len(trainset)):
            image, _, _ = trainset[i]  
            image_gpu = image.cuda() if torch.cuda.is_available() else image  # Move tensor to GPU if available
            image_vector = image_gpu.view(-1)  # Flatten the image tensor
            image_vectors.append(image_vector)
        
        image_matrix = torch.stack(image_vectors).cuda() if torch.cuda.is_available() else torch.stack(image_vectors)  # Convert to PyTorch tensor and move to GPU if available
        return image_matrix


    def images_to_matrix_lists(self) -> None:
        """
        Convert images in the dataset to matrix lists.
        """
        for _ in range(self.repetitions):
            image_matrix = self.images2matrix(self.dataset)
            self.images_matrix_lists.append(image_matrix)  # Append the matrix to the list
            
        # print(self.images_matrix_lists)

    def get_BOF(self) -> List[Dict[str, Dict[str, Any]]]:
        """
        Compute Bag of Features (BOF) from the image matrices.
        Returns:
        - List of dictionaries containing BOF features.
        """
        results = []
        for image_matrix in self.images_matrix_lists:
            # print(image_matrix.shape)
            # Perform operations using each matrix, such as Effective_Ranks
            get_rank = Effective_Ranks(image_matrix)
            r0 = get_rank.r0
            R0 = get_rank.R0
            rk_max_index = get_rank.rk_max_index
            rk_max = get_rank.rk_max_value
            Rk_max = get_rank.Rk_value_max_rk_index
            results.append({
                "isic": {
                    "r0": r0,
                    "R0": R0,
                    "rk_max_index": rk_max_index,
                    "rk_max": rk_max,
                    "Rk_max": Rk_max
                }
            })
        return results
    
    def calculate_stats(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate statistics (mean and standard deviation) from the BOF features.
        Returns:
        - Dictionary containing calculated statistics.
        """
        data_list = self.BOF_feature_list
        keys = data_list[0]['isic'].keys()  # Get keys
        results = {}
        for key in keys:
            values = [item['isic'][key] for item in data_list]  # Collect all values for a specific key
            mean = statistics.mean(values)  # Calculate mean
            std_dev = statistics.stdev(values)  # Calculate standard deviation
            results[key] = (mean, std_dev)  # Store mean and standard deviation as a tuple
        return results
    
    def calculate_stats_tensor(self) -> Dict[str, Tuple[float, float]]:
        """
        计算 BOF 特征的均值和标准差。
        返回：
        - 包含每个特征计算的均值和标准差的字典。
        """
        data_list = self.BOF_feature_list
        keys = data_list[0]['isic'].keys()

        results = {}
        for key in keys:
            values = [item['isic'][key] for item in data_list]
            values_tensor = torch.tensor(values, dtype=torch.float32)  # 转换为浮点数张量

            mean = torch.mean(values_tensor).item()  # 计算均值并转换为 Python 标量
            std_dev = torch.std(values_tensor).item()  # 计算标准差并转换为 Python 标量

            results[key] = (mean, std_dev)

        return results

    
    def save_stats_to_file(self, file_path: str) -> None:
        """
        Save computed statistics to a file.
        Args:
        - file_path: Path to save the statistics file.
        """
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(file_path, 'wb') as file:
            pickle.dump(self.BOF_mean_stddev, file)


class CompareBOF():
    # 这里是为了比较不同的增强强度下的某一种model的表现力

    def __init__(self,
                file_path: str,
                target_pkl: str,
                net_name: str='data',
                aug_name: str='angle'


                ) -> None:
        self.folder_path = file_path
        self.target_pkl = target_pkl

        self.matching_paths = []
        self.find_matching_pkls()

        self.comb_acc = []
        self.comb_acc_matrix = None
        # self.compare_acc()
        self.comb_BOF = []
        self.compare_BOF()

        # self.draw_BOF(net_name=net_name, aug_name=aug_name)
        


        # self.draw_acc(net_name=net_name, aug_name=aug_name/)
        
    def try_load_pkl(self, file_path: str) -> Union[Any, None]:
        """
        尝试加载一个Pickle文件。

        Args:
        - file_path (str): Pickle文件的路径

        Returns:
        - Union[Any, None]: 返回加载的数据，如果出现错误则返回None
        """
        try:
            # 使用绝对路径加载 Pickle 文件
            with open(file_path, 'rb') as file:
                data = pickle.load(file)    # 这里的文件的路径最好使用绝对路径，不然打不开
                # print(f"{file_path} Pickle file loaded successfully.")
                # 这里可以添加你想要做的事情，比如打印数据内容
                # print(data)
                return data
        except FileNotFoundError:
            print("File not found. Please provide a valid file path.")
        except Exception as e:
            print("An error occurred:", e)
    
    def find_matching_pkls(self)-> None:
        """
        在给定文件夹路径下搜索与输入的 pkl 文件名相匹配的 pkl 文件，并返回这些文件的路径列表。

        Args:
        - folder_path (str): 要搜索的文件夹路径
        - target_pkl (str): 目标 pkl 文件名

        Returns:
        - list: 匹配的 pkl 文件路径列表
        """
        # 遍历文件夹下的子文件夹
        for root, dirs, files in os.walk(self.folder_path):
            # print('1')
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # 检查子文件夹中的 pkl 文件是否与目标 pkl 文件名匹配
                pkl_files = [f for f in os.listdir(dir_path) if f.endswith('.pkl') and f == self.target_pkl]
                self.matching_paths.extend([os.path.join(dir_path, pkl_file) for pkl_file in pkl_files])
        def get_number_from_path(path):
            # 从路径中提取数字部分
            try:
                number = float(path.split('\\')[-2])
                if number.is_integer():
                    return int(number)
                else:
                    return number
            except ValueError:
                return None  # 或者返回适当的默认值或者标记

        self.matching_paths = sorted(self.matching_paths, key=get_number_from_path)
        # print(self.matching_paths)

    def compare_BOF(self):
        for pkl_path in self.matching_paths:
            # print(pkl_path)
            temp_get_BOF = self.try_load_pkl(file_path=pkl_path)
            # print(temp_get_acc.feature_cared)
            # value = list(temp_get_acc.values())[0]
            # print('{}'.format(value))
            self.comb_BOF.append(temp_get_BOF)
            # print(temp_get_BOF)
        # self.comb_acc_matrix = np.array(self.comb_acc)


            


    def draw_BOF(self, net_name, aug_name):
        save_path = os.path.join(self.folder_path, f'BOF_{net_name}_{aug_name}.png')
        excel_path = os.path.join(self.folder_path, f'BOF_{net_name}_{aug_name}.xlsx')
        data = self.comb_BOF

        # 创建4个子图的大图布局
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))  # 5个子图

        # 遍历每个子图的索引和对应的键
        keys = ['r0', 'R0', 'rk_max_index', 'rk_max', 'Rk_max']  # 五个键

        # Create DataFrames to store mean and error values
        mean_df = pd.DataFrame()
        error_df = pd.DataFrame()

        for idx, subkey in enumerate(keys):
            if aug_name == 'scale' or aug_name == 'crop':
                values = [item[subkey][0] for item in data[::-1]]
                errors = [item[subkey][1] for item in data[::-1]]
            else:
                values = [item[subkey][0] for item in data]
                errors = [item[subkey][1] for item in data]

            # Store data in DataFrames
            mean_df[subkey] = values
            error_df[subkey] = errors

            # 绘制子图带误差棒
            axs[idx].errorbar(np.arange(len(values)) / (len(values) - 1), values, yerr=errors, linestyle=':', marker='o', markersize=4)
            axs[idx].set_title(subkey)

        # 保存数据到Excel文件
        with pd.ExcelWriter(excel_path) as writer:
            mean_df.to_excel(writer, sheet_name='Mean', index=False)
            error_df.to_excel(writer, sheet_name='Error', index=False)

        # 调整布局并保存图像
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        plt.close()

   


if __name__ == '__main__':
    temp = ISICBOF(save_file_path=".\\Result\\isic_crop", costume_transform=None, crop=16)
    print(temp.BOF_mean_stddev)
