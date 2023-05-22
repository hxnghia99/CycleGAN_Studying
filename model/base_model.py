#=======================================================================================#
#                                                                                       #
#   File name   : base_model.py                                                         #
#   Author      : hxnghia99                                                             #
#   Created date: May 18th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : base class for all model types using for training and testing         #
#                                                                                       #
#=======================================================================================#

"""This module implements an abstract base class (ABC) 'BaseModel' for all models"""

import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
from collections import OrderedDict



class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        #Four obliged lists
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by function <optimize_parameters> during training and <test> during testing."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    @staticmethod
    def modify_commandline_options(parser, is_train: bool):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser
    

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')


    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        self.print_networks(opt.verbose)
    
    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('\nLearning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret




"""---------------------------------------SUPPORT FUNCTIONS AND CLASSES---------------------------------------------"""
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        - norm_type (str) : define the type of normalization layer as batch | instance | none
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True) #use learnable affine parameters and track running statistics (mean/stddev)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False) #NOT use learnable affine parameters and NOT track running statistics
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError("Normalization layer [%s] is not found" % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize the network weights
    
    Parameters:
        - net (network)         : the network to initialize
        - init_type (str)       : name of initialization method: normal | xavier | kaiming | orthogonal
        - init_gain (float)     : scaling factor for method: normal | xavior | orthogonal
    
    Note: 'normal' is the default option in original paper, but xavier and kaiming might work better in some applications
    """
    def init_func(m):
        classname = m.__class__.__name__
        #initialize all weights
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
            #initialize bias
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:           # BatchNorm Layer's weight is not a matrix --> only apply normal distribution 
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
        
        print("Initalize the network with %s" % init_type)
        net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network with two steps:
            1. Register CPU/GPU device (multi-GPU support)
            2. Initialize the network weights
        
    Parameters:
        - net (network)         : the network to initialize
        - init_type (str)       : name of initialization method: normal | xavier | kaiming | orthogonal
        - init_gain (float)     : scaling factor for method: normal | xavior | orthogonal
        - gpu_ids (list[int])   : specify the GPUs to run
    
    Returns:
        - an initialized network
    """
    if len(gpu_ids)>0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)   #multi-GPUs
    init_weights(net, init_type, init_gain)
    return net

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('Learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


"""---------------------------------------SUPPORT FUNCTIONS AND CLASSES---------------------------------------------"""









"""---------------------------------------GENERATOR---------------------------------------------"""
#Identity layer when not using normalization layer
class Identity(nn.Module):
    def forward(self, x):
        return x

class ResnetBlock(nn.Module):
    """Define a Resnet block (9 blocks)"""

    def __init__(self, dim, padding_type, norm_layer, use_bias):
        """Initalize the Resnet block
        
        Parameters:
            - dim (int)             : the number of channels in the conv layer
            - padding_type (str)    : the name of padding layer: reflect | replicate | zero
            - norm_layer            : normalization layer
            - use_bias (bool)       : whether conv layer uses bias
        
        Returns:
            - a Resnet block
        """
        super(ResnetBlock, self).__init__()

        conv_block = []
        padding = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding = 1
        else:
            raise NotImplementedError("Padding [%s] is not implemented" % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        
        padding = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            padding = 1
        else:
            raise NotImplementedError("Padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=padding, bias=use_bias),
                       norm_layer(dim)]
        
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, x):
        """Forward function (implementing the skip connection)"""
        out = x + self.conv_block(x)
        return out
    
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks (9-blocks) between a few downsampling/upsampling operations."""
    
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=9, padding_type='reflect'):
        """Construct a Resnet-based generator
        
        Parameters:
            - input_nc (int)        : the number of channels in input images
            - output_nc (int)       : the number of channels in output images
            - ngf (int)             : the number of filters in the last convolutional layer
            - norm_layer            : normalization layer: batch | instance
            - n_blocks (int)        : the number of Resnet blocks
            - padding_type (str)    : the name of padding layer in convolutional layers: reflect | replicate | zero
        """
        super(ResnetGenerator, self).__init__()
        
        #Use bias when using Instance Batch Normalization
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        """First conv layer"""
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        #Padding --> conv2d (7x7+bias) --> norm2d --> ReLU

        """2x Downsampling layers"""
        n_downsampling = 2
        for i in range(n_downsampling):
            multiplier = 2 ** i
            model += [nn.Conv2d(ngf * multiplier, ngf * multiplier * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * multiplier * 2),
                      nn.ReLU(True)]
            #conv2d (3x3+bias, stride=2) --> norm2d --> ReLU
        
        """9x Resnet blocks"""
        multiplier = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * multiplier, padding_type=padding_type, norm_layer=norm_layer, use_bias=use_bias)]
            #Padding+conv2d+ReLU + Padding+conv2d --> skip connection --> x9
        
        """2x Upsampling layers"""
        for i in range(n_downsampling):
            multiplier = 2 ** (n_downsampling - i) # output: 2 --> 1
            model += [nn.ConvTranspose2d(ngf * multiplier, int(ngf * multiplier / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                      norm_layer(int(ngf * multiplier /2)),
                      nn.ReLU(True)]
            #Transposed-conv2d (3x3+bias, stride=2) --> norm_layer --> ReLU

        """Last conv layer"""
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0, bias=use_bias),
                  nn.Tanh()]
        #Padding --> Conv2d (7x7+bias) --> Tanh

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Forward function"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection"""

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
        """Construct a Unet submodule with skip connections
        
        Parameters:
            - outer_nc (int)                        : the number of filters in the outer conv layer
            - inner_nc (int)                        : the number of filters in the inner conv layer
            - input_nc (int)                        : the number of channels in the input images 
            - submodule (UnetSkipConnectionBlock)   : previously defined submodules
            - outermost (bool)                      : whether this module is outermost module
            - innermost (bool)                      : whether this module is innermost module
            - norm_layer                            : normalization layer
        
        Returns:
            - Unet model"""
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        if input_nc is None:
            input_nc = outer_nc
        
        donwconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [donwconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, donwconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, donwconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward function"""
        if self.outermost:
            return self.model(x)
        else:   #innermost: add skip connections
            return torch.cat([x, self.model(x)], 1)
        
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc:int, output_nc:int, num_downs:int, ngf:int=64, norm_layer=nn.InstanceNorm2d):
        """Construct a Unet generator
        
        Parameters:
            - input_nc (int)            : the number of channels in input images
            - output_nc (int)           : the number of channels in output images
            - num_downs (int)           : the number of downsamplings in UNet, the image resolution must be a multiplier of 2**num_donws
            - ngf (int)                 : the number of filters in the last conv layer
            - norm_layer                : normalization layer

        Returns:
            - a Unet-based generator
        """
        super(UnetGenerator, self).__init__()

        #constuct Unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Forward function"""
        return self.model(input)

def define_G(input_nc: int, output_nc: int, ngf: int, netG: str, norm:str ='instance', init_type:str ='normal', init_gain:float =0.02, gpu_ids:list =[]):
    """Create a generator based on ResNet
    
    Parameters:
        - input_nc (int)                : the number of channels in input images
        - output_nc (int)               : the number of channels in output images
        - ngf (int)                     : the number of filters in the last conv layer
        - netG (str)                    : the architecture's name: resnet_9blocks | unet_256
        - norm (str)                    : the name of normalization layer: batch | instance| none
        - init_type (str)               : the name of initilization method
        - init_gain (float)             : scaling factor for normal, xavier and orthogonal
        - gpu_ids (list[int])           : specify the GPUs being used
    Returns:
        - a generator
    The generator is created by caller <init_net>
    """

    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, n_blocks=9)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)
"""---------------------------------------GENERATOR---------------------------------------------"""









"""---------------------------------------DISCRIMINATOR---------------------------------------------"""
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        
        Parameters:
            - input_nc (int)        : the number of channels in the input images
            - ndf (int)             : the number of filters in the first conv layer
            - n_layers (int)        : the number of conv layers in the discriminator
            - norm_layer            : normalization layer
        
        Returns:
            - a discriminator network
        """
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        kw = 4
        padw = 2
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
                    nn.LeakyReLU(0.2, True)]
        #conv(4x4+bias, stride=2) --> LeakyReLU: 1st downsampling + increase channels to ndf(64)
        
        nf_multiplier = 1
        nf_multiplier_previous = 1
        for n in range(1, n_layers):
            nf_multiplier_previous = nf_multiplier
            nf_multiplier = min(2**n, 8)                #maximum filters upto ndf(64)*8
            sequence += [nn.Conv2d(ndf*nf_multiplier_previous, ndf*nf_multiplier, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                         norm_layer(ndf * nf_multiplier),
                         nn.LeakyReLU(0.2, True)]
            # conv2d(4x4+bias, stride=2) + norm2D + LeakyReLU   x2
    
        nf_multiplier_previous = nf_multiplier
        nf_multiplier = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf*nf_multiplier_previous, ndf*nf_multiplier, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     norm_layer(ndf*nf_multiplier),
                     nn.LeakyReLU(0.2, True)]
        #Conv2d(4x4+bias) + norm2D + LeakyReLU
        """In total: reduce resolution by 8 and increase channels to ndf(64)*8"""
        
        sequence += [nn.Conv2d(ndf*nf_multiplier, 1, kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        """Forward fucntion"""
        return self.model(input)
    
class PixelDiscriminator(nn.Module):
    """Defines a PixelGAN discriminator"""
    
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a PixelGAN discriminator

        Parameters:
            - input_nc (int)    : the number of channels in input images
            ndf (int)           : the number of filters in the last conv layer
            norm_layer          : normalization layer
        
        Returns:
            - a PixelGAN discriminator
        """
        super(PixelDiscriminator, self).__init__()
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        #3xConv2d + 2xLeakyReLU + 1xnorm_layer
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

def define_D(input_nc:int, ndf:int, netD:str, n_layers_D:int=3, norm:str='batch', init_type:str='normal', init_gain:float=0.02, gpu_ids:list=[]):
    """Create a discriminator
    
    Parameters:
        - input_nc (int)            : the number of channels in input images
        - ndf (int)                 : the number of filters in the first conv layer
        - netD (str)                : specify architecture's name: basic (from PatchGAN) | n_layers | pixel
        - n_layers_D (int)          : the number of conv layers in discriminator if using <net_D>='n_layers'
        - norm (str)                : the type of normalization layers
        - init_type (str)           : the name of initialization method
        - init_gain (float)         : scaling factor of initialization method
        - gpu_ids (list[int])       : specify GPUs to run
    
    Returns:
        - an initialized discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)
"""---------------------------------------DISCRIMINATOR---------------------------------------------"""







"""---------------------------------------GAN-LOSS---------------------------------------------"""
class GANLoss(nn.Module):
    """Define GAN objective loss function"""

    def __init__(self, gan_obj, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str)              : the type of GAN objective loss: vanilla | lsgan
            target_real_label (bool)    : label for a real image
            target_fake_label (bool)    : label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))     #create buffer for label
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_obj = gan_obj
        if gan_obj == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_obj == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('Gan mode [%s] is not implemented' % gan_obj)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor)     : tpyically the prediction from a discriminator
            target_is_real (bool)   : if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)      #expand label into shape of prediction

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth label.

        Parameters:
            prediction (tensor)     : tpyically the prediction output from a discriminator
            target_is_real (bool)   : if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_obj in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss