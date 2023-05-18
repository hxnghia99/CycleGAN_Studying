#=======================================================================================#
#                                                                                       #
#   File name   : train_options.py                                                      #
#   Author      : hxnghia99                                                             #
#   Created date: May 18th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : setting the training-options for cycleGAN along with base-options     #
#                                                                                       #
#=======================================================================================#


from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training-options which are added to base-options"""

    def initialize(self, parser):
        parser = super().initialize(parser)
        # parser = BaseOptions.initialize(self, parser)
        
        parser.add_argument('--phase', type=str, default='train', help='the running phase: training, testing')

        return parser