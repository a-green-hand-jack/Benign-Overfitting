import os

import cv2
import numpy as np
import torch
import torch.utils.data
from typing import Callable
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, resize=512):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # print(self.img_dir, self.mask_dir)
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        mask = cv2.imread(os.path.join(self.mask_dir,img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None]
        if self.transform:
            img, mask = self.transform(img, mask)
        
        return img, mask, {'img_id': img_id}



def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images
    
class ImageToImage2D(torch.utils.data.Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False, resize:int = 128) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask
        self.resize = resize

        self.format_mapping = {'.jpg': '.png', '.png': '.png', '.jpeg': '.png'}  # Add more mappings if needed

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, idx):
        # Ensure idx does not exceed 2000
        # if idx >= 2000:
        #     raise IndexError("Index out of range for 2000 samples.")
        num_samples = len(self.images_list)
        if idx >= len(self.images_list):
            # raise IndexError("Index out of range for image list.")
            # Ensure idx stays within range
            idx = idx % num_samples

        image_filename = self.images_list[idx]
        image_extension = os.path.splitext(image_filename)[1].lower()
        
        # Check if the mask exists for the image
        # mask_filename = image_filename[:-len(image_extension)] + self.format_mapping.get(image_extension)
        mask_filename = image_filename[:-len(image_extension)] + ".png"
        if mask_filename not in os.listdir(self.output_path):
            # If mask doesn't exist, skip this sample and move to the next one
            return self.__getitem__(idx + 1)  # Skip to the next sample
        
        # Read image
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        
        # Read mask
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        
        # Adjust image and mask size to 128x128
        # image = cv2.resize(image, (self.resize, self.resize))
        # mask = cv2.resize(mask, (self.resize, self.resize), interpolation=cv2.INTER_NEAREST)
        
        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        
        # Correct dimensions if needed
        image, mask = correct_dims(image, mask)
        
        # Apply transformations if specified
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        # Convert mask to one-hot encoding if needed
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return image, mask, image_filename
    


# class JointTransform2D:
#     """
#     Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
#     it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
#     Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
#     be used, which will take care of the problems above.

#     Args:
#         crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
#             be taken.
#         p_flip: float, the probability of performing a random horizontal flip.
#         color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
#             If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
#         p_random_affine: float, the probability of performing a random affine transform using
#             torchvision.transforms.RandomAffine.
#         long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
#     """
#     def __init__(self, crop=(32, 32), p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
#                  p_random_affine=0, long_mask=True):
#         self.crop = crop
#         self.p_flip = p_flip
#         self.color_jitter_params = color_jitter_params
#         if color_jitter_params:
#             self.color_tf = T.ColorJitter(*color_jitter_params)
#         self.p_random_affine = p_random_affine
#         self.long_mask = long_mask

#     def __call__(self, image, mask):
#         # transforming to PIL image
#         image, mask = F.to_pil_image(image), F.to_pil_image(mask)

#         # random crop
#         if self.crop:
#             i, j, h, w = T.RandomCrop.get_params(image, self.crop)
#             image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

#         if np.random.rand() < self.p_flip:
#             image, mask = F.hflip(image), F.hflip(mask)

#         # color transforms || ONLY ON IMAGE
#         if self.color_jitter_params:
#             image = self.color_tf(image)

#         # random affine transform
#         if np.random.rand() < self.p_random_affine:
#             affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
#             image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

#         # transforming to tensor
#         image = F.to_tensor(image)
#         mask = F.to_tensor(mask)
#         # if not self.long_mask:
#         #     mask = F.to_tensor(mask)
#         # else:
#         #     mask = to_long_tensor(mask)
#         # Ensure the mask has the expected shape [1, H, W] for single-channel masks
#         # if len(mask.shape) > 2:
#             # mask = mask.unsqueeze(1)  # Add an extra dimension if needed

#         return image, mask
    
class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separately.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        augmentations: list of torchvision.transforms or None, list of augmentation transforms to be applied.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """
    def __init__(self, augmentations=None, long_mask=True):
        self.augmentations = augmentations
        self.long_mask = long_mask

    def __call__(self, image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # apply user-defined augmentations
        if self.augmentations:
            for augmentation in self.augmentations:
                image = augmentation(image)
                # Apply augmentation only to the image, not the mask
                if 'normalize' not in augmentation.__class__.__name__.lower():
                    mask = augmentation(mask)

        # transforming to tensor
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)

        # Calculate mean and std for current image and mask
        img_mean = image.mean(dim=[1, 2])
        img_std = image.std(dim=[1, 2])
        mask_mean = mask.mean()
        mask_std = mask.std()

        # Normalize both image and mask using calculated mean and std
        image = F.normalize(image, mean=img_mean, std=img_std+1e-7, inplace=True)
        mask = F.normalize(mask, mean=mask_mean, std=mask_std+1e-7, inplace=True)

        # Ensure the mask has the expected shape [1, H, W] for single-channel masks
        if len(mask.shape) > 2:
            mask = mask.unsqueeze(1)  # Add an extra dimension if needed

        return image, mask









