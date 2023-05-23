#=======================================================================================#
#                                                                                       #
#   File name   : unaligned_dataset.py                                                  #
#   Author      : hxnghia99                                                             #
#   Created date: May 18th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : Define a dataset class to generate unpaired images for CycleGAN       #
#                                                                                       #
#=======================================================================================#

import os
import random
from PIL import Image
import torchvision.transforms as transform
from .base_dataset import BaseDataset, make_dataset_path
from torchvision.ops import masks_to_boxes



class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path_to_dataset/trainA'
    and from domain B '/path_to_data/trainB' respectively.
    The path to dataset is configured with the dataset flag '--dataroot /path_to_data'
    """

    def __init__(self, opt):
        #Initialize the dataset
        super().__init__(opt)
        # BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')        #create path [dataroot]/trainA
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')        #create path [dataroot]/trainB

        self.A_paths = sorted(make_dataset_path(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset_path(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        BtoA_flag = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if BtoA_flag else self.opt.input_nc
        output_nc = self.opt.input_nc if BtoA_flag else self.opt.output_nc

    def __len__(self) -> int:
        """Return the total number of images in the dataset: maximum of two different number of images from two domain"""
        return max(self.A_size, self.B_size)
    
    def __getitem__(self, index: int) -> dict:
        """Return a data point including image and label
        
        Parameters:
            - index (int)           : a random integer for data indexing
            
        Returns:
            - A dictionary containing image A, image B, label A, label B
        """
        
        A_path = self.A_paths[index % self.A_size]  #make sure index is within range
        if self.opt.serial_batches:         
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        #Read images
        A_image = Image.open(A_path).convert("RGB")
        B_image = Image.open(B_path).convert("RGB")

        #Read labels
        A_label = A_path.split("jpg")[0] + "png"
        A_label = Image.open(A_label)
        B_label = B_path.split("jpg")[0] + "png"
        B_label = Image.open(B_label)

        #Apply identical transformation methods to both image and label
        A_image, A_label, A_box = self.data_transform(A_image, A_label)
        B_image, B_label, B_box = self.data_transform(B_image, B_label)

        return {'A_image': A_image, 'B_image': B_image, 'A_label': A_label, 'B_label': B_label, 'A_path': A_path, 'B_path':B_path, 'A_box':A_box, 'B_box': B_box}
    
    def data_transform(self, image, label, itp_method=transform.InterpolationMode.BICUBIC):
        #resize
        if 'resize' in self.opt.preprocess:
            osize = [self.opt.load_size, self.opt.load_size]
            resize = transform.Resize(osize, itp_method)
            image = resize(image)
            label = resize(label)
        
        #crop
        if 'crop' in self.opt.preprocess:
            osize = [self.opt.crop_size, self.opt.crop_size]
            i, j, h, w = transform.RandomCrop.get_params(image, output_size=osize) 
            image = transform.functional.crop(image, i, j, h, w)
            label = transform.functional.crop(label, i, j, h, w)

        #horizontal flip
        if 'hflip' in self.opt.preprocess:
            if random.random() < 0.5:
                image = transform.functional.hflip(image)
                label = transform.functional.hflip(label)
        
        #convert to tensor
        toTensor = transform.ToTensor()
        image = toTensor(image)
        label = toTensor(label)*255.0


        if label.max() == 0:
            box = [[0., 0., 0., 0.]]
        else:
            box = masks_to_boxes(label)
        # label[0,int(box[0][0]):int(box[0][2]),int(box[0][1]):int(box[0][3])] = 1

        #Normalize image
        normalize = transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = normalize(image)

        return image, label, box[0]