# 这里是之前发现MLP和LeNet的input计算得到的death len的趋势竟然不一样！！这一点实在是十分的奇怪，所以这里为了验证这个现象，我需要单独的计算一下data本身的betti number的情况

# 加载自己的包
from TDA.data_tda import Image2TDA, CompareTDA


# 加载其他库
import numpy as np
import torchvision.transforms as transforms

# 首先考虑scale作为增强

def scale_data_tda(scale_path = "./Result/DataTDA", aug_name="scale"):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{aug_name}"

    
    scale_list = np.arange(0.1, 1.02, 0.02)
    for scale in scale_list:
        
        train_transform=transforms.Compose([
                                    transforms.RandomResizedCrop(size=(32,32), scale=(scale, scale)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                                ])
        # train_loader, test_loader = get_cifar10_dataloader(batch_size=64, custom_transform=train_transform)

        print(f"裁切是{scale}.")

        save_floor = f"{scale_path}/{scale*10}/"

        temp_img = Image2TDA(costume_transform=train_transform, repetitions=10, save_file_path = save_floor)

def angle_data_tda(scale_path = "./Result/DataTDA", aug_name="angle"):
    # 这里我希望得到的是在某一个model下的在scale增强下的情况
    image_size = 32
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.2023, 0.1994, 0.2010]
    scale_path = f"{scale_path}/{aug_name}"

    
    min_angle_list = range(0,180,1)
    # for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
    for max_angle in min_angle_list:
        min_angle = -max_angle

        train_transform=transforms.Compose([
                            transforms.RandomRotation(degrees=(min_angle, max_angle)),
                            transforms.ToTensor(),
                            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                            ])
        # train_loader, test_loader = get_cifar10_dataloader(batch_size=64, custom_transform=train_transform)

        print(f"现在的最小角度是{min_angle}，最大角度是{max_angle}.")

        save_floor = f"{scale_path}/{max_angle}/"

        temp_img = Image2TDA(costume_transform=train_transform, repetitions=10, save_file_path = save_floor)
if __name__ == '__main__':

    # scale_data_tda()

    # angle_data_tda()
    

    folder = ".\\Result\\DataTDA\\angle"
    test_BOF = CompareTDA(file_path=folder, target_pkl="L2_betti_features.pkl")

    print(test_BOF.comb_BOF)
    test_BOF.draw_BOF(net_name="data", aug_name="angle")