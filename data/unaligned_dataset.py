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
import numpy as np
import torch
import json

from torchvision.utils import draw_bounding_boxes

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

        self.label_type = opt.label_type

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

        if self.label_type == "bbox":
            #A
            A_name = A_path.split("\\")[-1].split(".png")[0] if ".png" in A_path else A_path.split("\\")[-1].split(".jpg")[0]
            A_label_path = "/".join(A_path.split("\\")[0:-1]) + "/labels/" + A_name + ".json"
            A_label_info = json.load(open(A_label_path,'r'))
            xmin,ymin,xmax,ymax = A_label_info['bbox']
            size = A_label_info['width_height']
            
            A_label = np.zeros(size, dtype=np.uint8)
            A_label[xmin:xmax+1, ymin:ymax+1] = 1 #* 255
            A_label = Image.fromarray(A_label.T, mode="L")
            #B
            B_name = B_path.split("\\")[-1].split(".png")[0] if ".png" in B_path else B_path.split("\\")[-1].split(".jpg")[0]
            B_label_path = "/".join(B_path.split("\\")[0:-1]) + "/labels/" + B_name + ".json"
            B_label_info = json.load(open(B_label_path,'r'))
            xmin,ymin,xmax,ymax = B_label_info['bbox']
            size = B_label_info['width_height']
            
            B_label = np.zeros(size, dtype=np.uint8)
            B_label[xmin:xmax+1, ymin:ymax+1] = 1 #* 255
            B_label = Image.fromarray(B_label.T, mode="L")


            # A_image = transform.ToPILImage()(draw_bounding_boxes(transform.PILToTensor()(A_image), torch.tensor([A_label_info['bbox']]), width=5, colors='blue'))
            # A_image.show()
            # A_label.show()
            # print("Test")

            # B_image = transform.ToPILImage()(draw_bounding_boxes(transform.PILToTensor()(B_image), torch.tensor([B_label_info['bbox']]), width=5, colors='blue'))
            # B_image.show()
            # B_label.show()
            # print("Test")

            #Apply identical transformation methods to both image and label
            A_image, A_label, A_box = self.data_transform(A_image, A_label, A_path)
            B_image, B_label, B_box = self.data_transform(B_image, B_label, B_path)
            
        elif self.label_type == "segmentation": 
            #A
            A_name = A_path.split("\\")[-1].split(".png")[0] if ".png" in A_path else A_path.split("\\")[-1].split(".jpg")[0]
            A_label_path = "/".join(A_path.split("\\")[0:-1]) + "/labels/" + A_name + ".png"
            A_label = Image.open(A_label_path)
            #B
            B_name = B_path.split("\\")[-1].split(".png")[0] if ".png" in B_path else B_path.split("\\")[-1].split(".jpg")[0]
            B_label_path = "/".join(B_path.split("\\")[0:-1]) + "/labels/" + B_name + ".png"
            B_label = Image.open(B_label_path)

            #Apply identical transformation methods to both image and label
            A_image, A_label, A_box = self.data_transform(A_image, A_label, A_path)
            B_image, B_label, B_box = self.data_transform(B_image, B_label, B_path)

            # test = Image.fromarray(np.array(B_label[0].to('cpu'))*255.0)
            # test.show()
            # print("a")
        
        elif self.label_type == "None":
            A_label=None
            A_box=None

        else:
            raise NotImplementedError("Do not implement the label type for training...")
            
        return {'A_image': A_image, 'B_image': B_image, 'A_label': A_label, 'B_label': B_label, 'A_path': A_path, 'B_path':B_path, 'A_box':A_box, 'B_box': B_box}
        

    def data_transform(self, image, label, image_path, itp_method=transform.InterpolationMode.BICUBIC):
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
        label = toTensor(label)*255.0           #scale 0->1
        label.type(torch.uint8)

        if label.max()==0:
            raise NotImplementedError("Error when creating label in transform func()..........")

        box = masks_to_boxes(label)

        #Normalize image
        normalize = transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = normalize(image)

        return image, label, box[0]