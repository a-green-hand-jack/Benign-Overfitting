from post_process.betti_feature_compare import GetFeatureCared
import os
import matplotlib.pyplot as plt
import numpy as np

class CompareModel():
    def __init__(self,
                
                l_distance: str = 'L2',
                feature2get: str = 'all_bars_survive_time_sum',
                betti_number_dim: str = '1th',
                layer_care: bool = False,
                
                models: list=['MLP', 'LeNet', 'ResNet18', 'ResNet34', 'ResNet50'],
                folder_number=0,
                aug_type = "angle",
                save_path = "",
                file_path: str = None
                ) -> None:
        
        self.file_path = file_path
        self.l_distance = l_distance
        self.feature2get = feature2get
        self.betti_number_dim = betti_number_dim
        self.layer_care = layer_care
        self.folder_path = save_path
        self.folder_number = folder_number

        self.data = self.find_file_paths(models=models, folder_number=folder_number, pre_path=self.file_path)
        print(self.data)

        self.draw_betti(save_path=save_path,data=self.data, model_list=models, aug_type=aug_type)


    def find_file_paths(self, 
                        models: list=['MLP', 'LeNet', 'ResNet18', 'ResNet34', 'ResNet50'], 
                        folder_number=0, 
                        pre_path: str=r".\pre_process_outputs\angle_no_train"):

        feature_list = []
        for model in models:
            file_name = 'betti_features.pkl'
            folder_path = os.path.join(pre_path, model, str(folder_number), file_name)
            # print(folder_path)
            temp_betti_feature = GetFeatureCared(file_path=folder_path, l_distance=self.l_distance, feature2get=self.feature2get, betti_number_dim=self.betti_number_dim, layer_care=self.layer_care)
            # print(temp_betti_feature.feature_cared)
            feature_list.append(temp_betti_feature.feature_cared)

        return feature_list

    def draw_betti(self, save_path, data, model_list, aug_type="angle"):
        save_path = os.path.join(save_path, f"compare_{self.folder_number}")
        categories = set(point[0] for sublist in data for point in sublist)
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'yellow', 'lime', 'gold']
        markers = ['o', 's', '^', 'D', 'x', 'h', 'H', 'p', 'P', '8', '<', '>']

        for idx, cat in enumerate(categories):
            x_values = [model_list[i] for i, sublist in enumerate(data) if any(point[0] == cat for point in sublist)]
            y_values = [point[1] for sublist in data for point in sublist if point[0] == cat]
            
            if aug_type == "scale":
                x_axis = np.arange(len(x_values)) / 10 + 0.1
            elif aug_type == "angle":
                x_axis = np.arange(len(x_values))
            
            plt.plot(x_values, y_values, linestyle=':', linewidth=1, markersize=5, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], label=str(cat))

        plt.xlabel('Models')
        plt.ylabel('Feature')
        plt.xticks(rotation=45)
        plt.title(f'{self.l_distance}_{self.feature2get}_{self.betti_number_dim}_{self.layer_care}')
        plt.legend()
        
        # 如果文件夹不存在，创建文件夹
        # directory = os.path.dirname(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 保存图片
        save_path = os.path.join(save_path, f'{self.l_distance}_{self.feature2get}_{self.betti_number_dim}_{self.layer_care}')
        plt.savefig(save_path)
        plt.show()








































