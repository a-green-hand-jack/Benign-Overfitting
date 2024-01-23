from BOF.cifar10_BOF import ImageProcessor
import albumentations as A
import numpy as np
import torchvision.transforms as transforms
import torch
# import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '512'


from BOF.cifar10_BOF import ImageProcessor, CompareBOF
image_size = 32
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]



def process_with_scale(scale_list=np.arange(0.1, 1.02, 0.02), input_height=32, input_width=32):
    for scale in scale_list:
        print("="*10, scale, "="*10, "\n")
        save_pkl_path = f"./cifar10/data_torch_gup/scale/{scale}/bof.pkl"
        # height = int(scale * input_height)
        # width = int(scale * input_width)
        # train_transform = A.Compose([
        #     A.RandomCrop(height=height, width=width, p=1.0),  # 随机裁剪
        #     A.Resize(32, 32),  # 调整大小
        #     A.Normalize(),  # 标准化
        # ])
        train_transform=transforms.Compose([
                                   transforms.RandomResizedCrop(size=(32,32), scale=(scale, scale)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                               ])
                                  
        
        temp_image_processor = ImageProcessor(save_file_path=save_pkl_path, costume_transform=train_transform, repetitions=10)
        # 清空GPU显存
        torch.cuda.empty_cache()
        print(temp_image_processor.BOF_mean_stddev)

    folder = "./Result/cifar10/data_torch/scale"
    test_BOF = CompareBOF(file_path=folder, target_pkl="bof.pkl", aug_name='scale')

    print(test_BOF.comb_BOF)
    test_BOF.draw_BOF(net_name="data", aug_name="scale")




def process_with_angle(min_angle_list=range(120, 180, 1)):
    for max_angle in min_angle_list:
        save_pkl_path = f"./cifar10_A/data_torch_gpu/angle/{max_angle}/bof.pkl"
        print(max_angle)
        min_angle = -max_angle

        train_transform={'train':transforms.Compose([
                            transforms.RandomRotation(degrees=(min_angle, max_angle)),
                            transforms.ToTensor(),
                            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
                            ])}
        train_transform = train_transform["train"]

        temp_image_processor = ImageProcessor(save_file_path=save_pkl_path, costume_transform=train_transform, repetitions=10)
        # 清空GPU显存
        torch.cuda.empty_cache()

    folder = "./cifar10_A/data_torch_gup/angle"
    test_BOF = CompareBOF(file_path=folder, target_pkl="bof.pkl", aug_name='scale')

    print(test_BOF.comb_BOF)
    test_BOF.draw_BOF(net_name="data", aug_name="angle")



if __name__ == '__main__':
    # main()
    process_with_angle()
    # process_with_scale()
