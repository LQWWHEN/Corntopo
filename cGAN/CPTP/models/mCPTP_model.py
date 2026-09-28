import torch
import torch.nn as nn

from .base_model import BaseModel
from . import networks
from .Losses import GradientLoss
import torch.nn.functional as F
class mCPTPModel(BaseModel):

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
                "--lambda_reg",
                type=float,
                default=5.0
            )
        parser.add_argument(
                    "--outer_activation",
                    type=str,
                    default="tanh"
                )


        return parser

    def fuse_attention(self, feats):

        pooled = []
        for f in feats:
            pooled.append(F.adaptive_avg_pool2d(f, 1).flatten(1))

        proj = []
        for i, f in enumerate(pooled):
            proj.append(self.proj_layers[i](f))  # (B, 128)

        stack = torch.stack(proj, dim=1)  # (B, S, 128)

        attn_logits = self.attn_fc(stack).squeeze(-1)  # (B, S)
        attn = torch.softmax(attn_logits, dim=1)

        fused = torch.sum(stack * attn.unsqueeze(-1), dim=1)

        return fused
    
    def __init__(self, opt):

        super().__init__(opt)

        self.loss_names = [
            "G_GAN",
            "G_L1",
            "G_Grad",
            "G_reg",
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
        self.condition = True if len(opt.conditions)>0 else False 
        self.pred_targets = True if len(opt.targets)>0 else False 
        if opt.outer_activation == "tanh":
            self.outer_activation = nn.Tanh()
        elif opt.outer_activation == "sigmoid":
            self.outer_activation = nn.Sigmoid()

        if self.isTrain:
            self.model_names = ["G", "D"]
        else:
            self.model_names = ["G"]

        self.netG = networks.define_G(
            input_nc=opt.input_nc,
            output_nc=opt.output_nc,
            ngf=opt.ngf,
            netG=opt.netG,
            norm=opt.norm,
            use_dropout=not opt.no_dropout,
            init_type=opt.init_type,
            init_gain=opt.init_gain,
            outer_activation=self.outer_activation,
            cond_dim= len(opt.conditions) if len(opt.conditions)>0 else None
        )

        self.proj_layers = None
        self.num_scales = None
        if self.pred_targets:
                
            self.attn_fc = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ).to(self.device)
            self.reg_head =  nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(self.opt.targets))
                ).to(self.device)
            

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
            self.criterionReg = nn.L1Loss()
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
        
        if self.condition:
            self.cond = batch["condition"].to(self.device)
        if self.pred_targets:
            self.targets = batch["targets"].to(self.device)
        self.label = batch["label"]


    def forward(self):
        if self.condition:
            self.fake_B,self.feats = self.netG(self.real_A,self.cond)
        else:
            self.fake_B,self.feats = self.netG(self.real_A)
        self.error = torch.abs(self.fake_B - self.real_B).detach()
        if self.pred_targets:
            if self.proj_layers is None:
                self.proj_layers = nn.ModuleList([
                nn.Linear(f.shape[1], 128).to(self.device)
                for f in self.feats
                ])
            self.reg_feat = self.fuse_attention(self.feats)
            self.pred_out = self.reg_head(self.reg_feat)
           
 

  

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
        mask_inter = self.mask_pre * self.mask_post  
        mask_inter = mask_inter.clamp(min=1e-6)
               # Soft weight using real mask
        weight = 1.0 + mask_inter  # mask=0 -> 1, mask=1 -> 2

        # Weighted L1
        weighted_error = torch.abs(self.real_B - self.fake_B) * weight
        self.loss_G_L1 = weighted_error.mean() * self.opt.lambda_L1

        # Weighted gradient loss
        self.loss_G_Grad = self.criterionGrad(
            self.fake_B,
            self.real_B,
            mask=mask_inter
        ) * self.opt.lambda_grad

        self.loss_G_reg = 0.0

        if self.pred_targets:
            self.loss_G_reg = self.criterionReg(
                self.pred_out,
                self.targets
            )

        self.loss_G = (
            self.loss_G_GAN +
            self.loss_G_L1+
            self.loss_G_Grad+
            self.loss_G_reg
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