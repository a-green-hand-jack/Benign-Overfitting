from UNeXt_TDA.val import UNeXtTDA
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms


def crop_Unext_parallel(dataset_name="busi", aug_name="crop", repetitions=10):

    
    if dataset_name == "BUSI":
        crop_list = range(8, 257, 8)
    elif dataset_name == "ISIC2018" or "MONU":
        crop_list = range(8, 513, 8)
    # for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
    def process_angle(crop=0.1):
        temp_img = UNeXtTDA(repetitions=repetitions, save_file_path = f'.\\Result\\UNext_output_TDA\\input_{dataset_name}\\{crop}', betti_dim=1, crop=crop, img_ext='.png',input_images_path = f'..\\..\\others_work\\dataset\\{dataset_name}\\train_folder\\images',
            input_masks_path = f'..\\..\\others_work\\dataset\\{dataset_name}\\train_folder\\masks')
        print(temp_img.feature2save)
    
     # 使用 ThreadPoolExecutor 实现并行计算
    with ThreadPoolExecutor() as executor:
        executor.map(process_angle, crop_list)

def crop_Unext(dataset_name="busi", aug_name="crop", repetitions=10):

    
    if dataset_name == "BUSI":
        crop_list = range(8, 257, 8)
    elif dataset_name == "ISIC2018" or "MONU":
        crop_list = range(8, 513, 8)
    # for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
    def process_angle(crop=0.1):
        temp_img = UNeXtTDA(repetitions=repetitions, save_file_path = f'.\\Result\\UNext_input_TDA\\input_{dataset_name}\\{crop}', betti_dim=1, crop=crop, img_ext='.png',input_images_path = f'..\\..\\others_work\\dataset\\{dataset_name}\\train_folder\\images',
            input_masks_path = f'..\\..\\others_work\\dataset\\{dataset_name}\\train_folder\\masks')
        # print(temp_img.feature2save)
    
    for crop in crop_list:
        print('='*10, '现在的dataset={}, crop={}'.format(dataset_name, crop), '='*10)
        process_angle(crop)

def process_with_angle(dataset_name="MONU" , repetitions=10, aug_name="crop"):

    max_angle_list = range(1,180,1)
    # for min_angle in tqdm(min_angle_list, unit="degree", desc="min_angle"):
    for max_angle in max_angle_list:
        min_angle = -max_angle
        print(f'最大的角度是{max_angle}, 研究的数据集是{dataset_name}')

        costume_transform = [transforms.RandomRotation(degrees=(min_angle, max_angle))]

        temp_image_processor = UNeXtTDA(repetitions=repetitions, 
                                        save_file_path = f'.\\Result\\UNext_output_TDA\\output_{dataset_name}{aug_name}\\{max_angle}', betti_dim=1,  img_ext='.png',
                                        costume_transform=costume_transform,
                                        input_images_path = f'..\\..\\others_work\\dataset\\{dataset_name}\\train_folder\\images',input_masks_path = f'..\\..\\others_work\\dataset\\{dataset_name}\\train_folder\\masks')

if __name__ == "__main__":
    # crop_Unext_parallel(dataset_name="busi", repetitions=10, aug_name="crop")
    # crop_Unext(dataset_name="BUSI", aug_name="crop", repetitions=10)
    # crop_Unext(dataset_name="ISIC2018", aug_name="crop", repetitions=2)
    # process_with_angle(dataset_name='BUSI', aug_name='Angle')
    process_with_angle(dataset_name='ISIC2018', aug_name='Angle')