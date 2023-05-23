#=======================================================================================#
#                                                                                       #
#   File name   : cycle_gan_model.py                                                    #
#   Author      : hxnghia99                                                             #
#   Created date: May 18th, 2023                                                        #
#   GitHub      : https://github.com/hxnghia99/CycleGAN_Styding                         #
#   Description : Define CycleGAN class containing training process and object loss     #
#                                                                                       #
#=======================================================================================#


from .base_model import BaseModel, define_G, define_D, GANLoss
from utils.image_pool import ImagePool
import torch
import itertools
from torchvision.ops import masks_to_boxes

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model for unpaired image-to-image translation.

    Options need noticing:
        --dataset_mode:     must be 'unaligned' to use unpaired dataset
        --netG:             set 'resnet_9blocks' for ResNet generator
        --netD:             set 'basic' to use discriminator from PatchGAN
        --gan_mode:         set 'lsgan' to use a least-squared objective function
    """
    def __init__(self, opt):
        """Initialize the CycleGAN class"""
        super().__init__(opt)
        # BaseModel.__init__(self, opt)

        # loss types to print out --> caller <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A','D_B', 'G_B', 'cycle_B', 'idt_B']
        
        #select image to save/display --> caller <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:         #add to visualizations if there is identity_loss
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B

        #select models to save --> caller <BaseModel.save_networks> in training and <BaseModel.load_networks> in testing
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']        
        else:
            self.model_names = ['G_A', 'G_B']

        # define networks: Generators and Discriminators
        self.netG_A = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.netD_A = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        
        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)   #identity loss only works when input and output have the same number of channels
            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size) 
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN   = GANLoss(opt.gan_obj).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt   = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A_image' if AtoB else 'B_image'].to(self.device)
        self.real_B = input['B_image' if AtoB else 'A_image'].to(self.device)

        self.label_A = input['A_label' if AtoB else 'B_label'].to(self.device)
        self.label_B = input['B_label' if AtoB else 'A_label'].to(self.device)

        self.box_A = input['A_box' if AtoB else 'B_box'].to(self.device)
        self.box_B = input['B_box' if AtoB else 'A_box'].to(self.device)

        # bbox_label_A = masks_to_boxes(self.label_A[0])[0]
        # self.label_A
        # bbox_label_B = masks_to_boxes(self.label_B[0])[0]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)*self.label_A + self.real_A*(1-self.label_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)*self.label_A +  self.fake_B*(1-self.label_A)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)*self.label_B + self.real_B*(1-self.label_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)*self.label_B +  self.fake_A*(1-self.label_B)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B * self.label_B, fake_B * self.label_A) * 10

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A * self.label_A, fake_A * self.label_B) * 10

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)*self.label_B + self.real_B*(1-self.label_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A * self.label_B, self.real_B * self.label_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)*self.label_A + self.real_A*(1-self.label_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B * self.label_A, self.real_A * self.label_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B * self.label_A), True) * 10
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A * self.label_B), True) * 10
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A * self.label_A, self.real_A * self.label_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B * self.label_B, self.real_B * self.label_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


    @staticmethod
    def modify_commandline_options(parser, is_train: bool):
        """Add new model-specific options, and rewrite default values for existing options (details in BaseModel)"""
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1000.0, help='weight for cycle loss A->B->A')
            parser.add_argument('--lambda_B', type=float, default=1000.0, help='weight for cycle loss B->A->B')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='weight for identity loss: A <-> G_B-A(A) and B <-> G_A-B(B)')
        
        return parser


    