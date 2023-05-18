#=======================================================================================#
#                                                                                       #
#   File name   : __init__.py                                                           #
#   Author      : hxnghia99                                                             #
#   Created date: May 18th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : init the datatypes for model and some fundamental functions           #
#                                                                                       #
#=======================================================================================#

"""This package includes all the modules related to data loading and preprocessing

To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
"""
import sys
sys.dont_write_bytecode = True
import importlib
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_mode: str):
    """Import the module "data/[dataset_name]_dataset.py"""
    
    dataset_filename = "data." + dataset_mode + "_dataset"          #data.[dataset_mode]_dataset, ex: data.unaligned_data
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_mode.replace("_","") + "dataset"  #unaligneddataset
    #search for all functions and classes in lib_file
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() and issubclass(cls, BaseDataset):
            dataset = cls
    
    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class_name matching %s in lowercase" % (dataset_filename, target_dataset_name))
    
    #return the found class of dataset
    return dataset 

def get_option_setter(dataset_mode: str):
    """ Return the static method <modify_commandline_options> of the dataset class"""
    dataset_class = find_dataset_using_name(dataset_mode)
    return dataset_class.modify_commandline_options