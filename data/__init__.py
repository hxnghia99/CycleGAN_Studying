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

import importlib
from data.base_dataset import BaseDataset
import torch.utils.data

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


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

    
class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""
    def __init__(self, opt):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("Dataset [%s] was created" % type(self.dataset).__name__)

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle= not opt.serial_batches,
                                                      num_workers=int(opt.num_threads))
    
    def load_data(self):
        return self
    
    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data      #suspend func's execution and send a value back to caller
