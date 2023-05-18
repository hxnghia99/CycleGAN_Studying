#=======================================================================================#
#                                                                                       #
#   File name   : test_options.py                                                       #
#   Author      : hxnghia99                                                             #
#   Created date: May 18th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : setting the testing-options for cycleGAN along with base-options      #
#                                                                                       #
#=======================================================================================#


from .base_options import BaseOptions

class TestOptions(BaseOptions):
    """This class includes testing-options which are added to base-options"""

    def initialize(self, parser):
        parser = super().initialize(parser)
        # parser = BaseOptions.initialize(self, parser)
        
        parser.add_argument('--phase', type=str, default='test', help='the running phase: training, testing')

        return parser