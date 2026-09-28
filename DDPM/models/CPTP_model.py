import torch
import torch.nn as nn

from .base_model import BaseModel
from . import networks
from .Losses import GradientLoss

class CPTPModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(
            netG="unet_128",
            netD="basic",
            norm="batch"
        )

        if is_train:
            parser.add_argument(
                "--lambda_L1",
                type=float,
                default=10.0
            )
            parser.add_argument(
                "--lambda_grad",
                type=float,
                default=5.0
            )
        parser.add_argument(
                    "--outer_activation",
                    type=str,
                    default="tanh"
                )
        

        return parser

    def __init__(self, opt):

        super().__init__(opt)

        self.loss_names = [
            "G_GAN",
            "G_L1",
            "G_Grad",
            "D_real",
            "D_fake"
        ]

        self.visual_names = [
            "real_A",
            "fake_B",
            "real_B",
            "error"
        ]
        self.outer_activation = nn.Tanh()

        if opt.outer_activation == "tanh":
            self.outer_activation = nn.Tanh()
        elif opt.outer_activation == "sigmoid":
            self.outer_activation = nn.Sigmoid()

        if self.isTrain:
            self.model_names = ["G", "D"]
        else:
            self.model_names = ["G"]

        # ----------------------------------
        # Generator
        # ----------------------------------

        self.netG = networks.define_G(
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            ngf=opt.ngf,
            netG=opt.netG,
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            outer_activation=self.outer_activation
        )

        # ----------------------------------
        # Discriminator
        # ----------------------------------

        if self.isTrain:

            self.netD = networks.define_D(
                input_nc=opt.input_nc + opt.output_nc,
                ndf=opt.ndf,
                netD=opt.netD,
                n_layers_D=opt.n_layers_D,
                norm=opt.norm,
                init_type=opt.init_type,
                init_gain=opt.init_gain
            )

        # ----------------------------------
        # Loss
        # ----------------------------------

        if self.isTrain:

            self.criterionGAN = networks.GANLoss(
                opt.gan_mode
            ).to(self.device)

            self.criterionL1 = nn.L1Loss()

            self.criterionGrad = GradientLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )

            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, 0.999)
            )

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, batch):

        self.real_A = batch["input"].to(self.device)
        self.real_B = batch["target"].to(self.device)

        self.mask_pre = batch["mask_pre"].to(self.device)
        self.mask_post = batch["mask_post"].to(self.device)

        self.label = batch["label"]


    def forward(self):

        self.fake_B = self.netG(self.real_A)
        self.error = torch.abs(self.fake_B - self.real_B).detach()

    def backward_D(self):

        fake_AB = torch.cat(
            [self.real_A, self.fake_B.detach()],
            dim=1
        )

        pred_fake = self.netD(fake_AB)

        self.loss_D_fake = self.criterionGAN(
            pred_fake,
            False
        )

        real_AB = torch.cat(
            [self.real_A, self.real_B],
            dim=1
        )

        pred_real = self.netD(real_AB)

        self.loss_D_real = self.criterionGAN(
            pred_real,
            True
        )

        self.loss_D = (
            self.loss_D_real +
            self.loss_D_fake
        ) * 0.5
 
        self.loss_D.backward()


    def backward_G(self):

        fake_AB = torch.cat(
            [self.real_A, self.fake_B],
            dim=1
        )

        pred_fake = self.netD(fake_AB)



        self.loss_G_GAN = self.criterionGAN(
            pred_fake,
            True
        )
      
        self.loss_G_L1 = (
            self.criterionL1(
                self.fake_B,
                self.real_B
            )
            * self.opt.lambda_L1
        )
        self.loss_G_Grad = (
            self.criterionGrad(
                self.fake_B,
                self.real_B
            )* self.opt.lambda_grad)
        self.loss_G = (
            self.loss_G_GAN +
            self.loss_G_L1+
            self.loss_G_Grad
        )
  
        self.loss_G.backward()

    
    def optimize_parameters(self):

        self.forward()

        # -------------------------
        # Update D
        # -------------------------

        self.set_requires_grad(
            self.netD,
            True
        )

        self.optimizer_D.zero_grad()

        self.backward_D()

        self.optimizer_D.step()

        # -------------------------
        # Update G
        # -------------------------

        self.set_requires_grad(
            self.netD,
            False
        )

        self.optimizer_G.zero_grad()

        self.backward_G()

        self.optimizer_G.step()