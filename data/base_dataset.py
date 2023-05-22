#=======================================================================================#
#                                                                                       #
#   File name   : base_dataset.py                                                       #
#   Author      : hxnghia99                                                             #
#   Created date: May 18th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : base class for all dataset types using for training and testing       #
#                                                                                       #
#=======================================================================================#

"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets"""

import torch.utils.data as data
from abc import ABC, abstractmethod
import os

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <get_transform>:                 (optionally) add transformation methods to each data (apply to both image and label)
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """
    
    def __init__(self, opt):
        """Initialize the class
        
        Parameters:
            - opt (class Options): store flags as well as settings for class creation
        """
        self.opt = opt
        self.root = opt.dataroot
    
    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset"""
        return 0
    
    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Return a data point including image, image_path, image_label
        
        Parameters:
            - index : a random integer for data selection
        
        Returns:
            - a dictionary of data
        """
        pass

    @abstractmethod
    def data_transform(self, image, label, itp_method):
        """Define the identical transformation for both image and label
        
        Parameters:
            - img:          input image
            - label:        segmentation label or bbox label
            - itp_method:   interpolation method when reszing image

        Returns:
            - img:          ouput image after data transformation
            - label:        segmentation label after data transformation
        """
        pass

    @staticmethod
    def modify_commandline_options(parser, is_train: bool):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser
    


#Check wether the file is an image
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
    #'.png', '.PNG',
]
def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#Save all directories of images into a list
def make_dataset_path(dir: str, max_dataset_size=float("inf")) -> list:
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    image_paths = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                image_path = os.path.join(root, fname)
                image_paths.append(image_path)
    return image_paths[:min(max_dataset_size, len(image_paths))]
