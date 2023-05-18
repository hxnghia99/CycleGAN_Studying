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

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
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
    def __getitem__(self, index: int):
        """Return a data point including image, image_path, image_label
        
        Parameters:
            - index : a random integer for data selection
        
        Returns:
            - a dictionary of data
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