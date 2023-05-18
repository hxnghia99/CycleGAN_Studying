#=======================================================================================#
#                                                                                       #
#   File name   : main.py                                                               #
#   Author      : hxnghia99                                                             #
#   Created date: May 18th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : the main code that create and train model based on settings           #
#                                                                                       #
#=======================================================================================#

""" General-purpose training script for image-to-image translation based on CycleGAN
    Change the options in options/base_options.py and options/train_options.py  """

from options.train_options import TrainOptions


opt = TrainOptions().parse()