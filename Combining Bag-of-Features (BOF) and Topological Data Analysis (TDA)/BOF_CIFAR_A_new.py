from BOF.cifar10_BOF import ImageProcessor
import albumentations as A
import numpy as np

from BOF.cifar10_BOF import ImageProcessor, CompareBOF



def process_with_scale(scale_list=np.arange(0.1, 1.02, 0.02), input_height=32, input_width=32):
    for scale in scale_list:
        save_pkl_path = f".\\cifar10_A\\data\\scale\\{scale}\\bof.pkl"
        height = int(scale * input_height)
        width = int(scale * input_width)
        train_transform = A.Compose([
            A.RandomCrop(height=height, width=width, p=1.0),  # 随机裁剪
            A.Resize(32, 32),  # 调整大小
            A.Normalize(),  # 标准化
        ])
        
        temp_image_processor = ImageProcessor(save_file_path=save_pkl_path, costume_transform=train_transform, repetitions=10)

    folder = ".\\cifar10_A\\data\\scale"
    test_BOF = CompareBOF(file_path=folder, target_pkl="bof.pkl", aug_name='scale')

    print(test_BOF.comb_BOF)
    test_BOF.draw_BOF(net_name="data", aug_name="scale")




def process_with_angle(min_angle_list=range(150, 181, 1)):
    for max_angle in min_angle_list:
        save_pkl_path = f".\\cifar10_A\\data\\angle\\{max_angle}\\bof.pkl"
        train_transform = A.Compose([
            # A.RandomCrop(height=height, width=width, p=1.0),  # 随机裁剪
            A.Rotate(limit=max_angle),  # 角度旋转增强，可设置旋转角度的限制
            A.Resize(32, 32),  # 调整大小
            A.Normalize(),  # 标准化
        ])

        temp_image_processor = ImageProcessor(save_file_path=save_pkl_path, costume_transform=train_transform, repetitions=10)

    folder = ".\\cifar10_A\\data\\angle"
    test_BOF = CompareBOF(file_path=folder, target_pkl="bof.pkl", aug_name='scale')

    print(test_BOF.comb_BOF)
    test_BOF.draw_BOF(net_name="data", aug_name="angle")



if __name__ == '__main__':
    # main()
    process_with_angle()
