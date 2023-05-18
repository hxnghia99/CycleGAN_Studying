#=======================================================================================#
#                                                                                       #
#   File name   : __init__.py                                                           #
#   Author      : hxnghia99                                                             #
#   Created date: May 18th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : init the model types and some fundamental functions                   #
#                                                                                       #
#=======================================================================================#

"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network.

Now you can use the model class by specifying flag '--model dummy'.
"""

import sys 
sys.dont_write_bytecode = True
import importlib
from model.base_model import BaseModel

def find_model_using_name(model_name: str):
    """Import the module "model/[model_name]_model.py"""

    model_filename = "model." + model_name + "_model"           #model.[model_name]_model, ex: model.cycle_gan_model
    modellib = importlib.import_module(model_filename)
    
    model = None
    target_model_name = model_name.replace("_","") + "model"              #cycleganmodel
    #search for all functions and classes in lib_file
    for name,cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseModel with class_name matching %s in lowercase" % (model_filename, target_model_name))
    
    #return the found class of model
    return model

def get_option_setter(model_name: str):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options
