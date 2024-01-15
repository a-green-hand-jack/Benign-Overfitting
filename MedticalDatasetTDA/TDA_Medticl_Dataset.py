from UNeXt_TDA.val import UNeXtTDA
from concurrent.futures import ThreadPoolExecutor


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
        temp_img = UNeXtTDA(repetitions=repetitions, save_file_path = f'.\\Result\\UNext_output_TDA\\input_{dataset_name}\\{crop}', betti_dim=1, crop=crop, img_ext='.png',input_images_path = f'..\\..\\others_work\\dataset\\{dataset_name}\\train_folder\\images',
            input_masks_path = f'..\\..\\others_work\\dataset\\{dataset_name}\\train_folder\\masks')
        # print(temp_img.feature2save)
    
    for crop in crop_list:
        process_angle(crop)


if __name__ == "__main__":
    # crop_Unext_parallel(dataset_name="busi", repetitions=10, aug_name="crop")
    crop_Unext(dataset_name="BUSI", aug_name="crop", repetitions=10)