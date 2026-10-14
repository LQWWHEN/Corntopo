"""Registered cGAN modules; optimization/checkpoints are owned by the CV runner."""
import torch
from torch import nn
from torch.nn import functional as F
from . import networks
from .Losses import GradientLoss, masked_mean


class MultiScaleSIAHead(nn.Module):
    def __init__(self, channels, hidden=128, output_dim=2):
        super().__init__()
        self.proj_layers = nn.ModuleList([nn.Linear(c, hidden) for c in channels])
        self.attn_fc = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.reg_head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, output_dim))

    def forward(self, feats):
        if len(feats) != len(self.proj_layers):
            raise ValueError('Unexpected generator feature scales')
        stack = torch.stack([proj(F.adaptive_avg_pool2d(f, 1).flatten(1))
                             for proj, f in zip(self.proj_layers, feats)], 1)
        weights = self.attn_fc(stack).softmax(dim=1)
        return self.reg_head((stack * weights).sum(1))


class BaseModel(nn.Module):
    def __init__(self, args, stats, training=True):
        super().__init__()
        self.args = args
        clinical_dim = len(args.conditions) if args.fusion_method == 'film' else None
        activation = {'identity': nn.Identity, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}[args.outer_activation]()
        self.netG = networks.define_G(len(args.input_maps), len(args.output_maps), args.ngf,
            args.netG, activation, norm=args.norm, use_dropout=args.dropout,
            cond_dim=clinical_dim, clinical_embed_dim=args.clinical_embed_dim)
        self.netSIA = (MultiScaleSIAHead(self.netG.feature_channels, args.sia_hidden_dim,
                                      output_dim=1 if args.sia_mode == 'scalar' else 2)
                       if args.sia_mode != 'off' else None)
        if args.sia_mode == 'vector':
            sia = stats['sia']
            mean = torch.tensor(sia['mean'], dtype=torch.float32)
            std = torch.tensor(sia['std'], dtype=torch.float32)
            if mean.shape != (2,) or std.shape != (2,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError('Invalid training-fold SIA statistics')
            self.register_buffer('sia_mean', mean)
            self.register_buffer('sia_std', std)
        # Retain D in inference checkpoints for strict round-trip loading.
        self.netD = networks.define_D(len(args.input_maps) + len(args.output_maps), args.ndf,
                                      args.netD, args.n_layers_D, args.norm)
        self.criterionGAN = networks.GANLoss(args.gan_mode)
        self.criterionGrad = GradientLoss()
        for net in (self.netG, self.netSIA, self.netD):
            if net is not None:
                networks.init_weights(net, args.init_type, args.init_gain)

    def generator_parameters(self):
        yield from self.netG.parameters()
        if self.netSIA is not None:
            yield from self.netSIA.parameters()

    def forward(self, pre, clinical=None):
        # No post map, post mask, or SIA ground truth can enter inference.
        pred, feats = self.netG(pre, clinical if self.args.fusion_method == 'film' else None)
        out = dict(pred=pred)
        if self.netSIA is not None:
            z = self.netSIA(feats)
            if self.args.sia_mode == 'scalar':
                value = z.squeeze(-1)
                if self.args.scalar_activation == 'softplus':
                    value = F.softplus(value)
                out['sia'] = value * self.args.sia_ref
            else:
                q = self.sia_mean + self.sia_std * z
                out.update(sia_state=z, sia_vector=q, sia=torch.linalg.vector_norm(q, dim=-1))
        return out

    @torch.no_grad()
    def sample(self, pre, clinical=None):
        return self(pre, clinical)

    def discriminator_input(self, pre, candidate, mask):
        # GT mask is used only to define the training loss domain, never generation.
        if self.args.reconstruction_domain == 'valid':
            candidate = candidate * mask
        return torch.cat((pre, candidate), 1)

    def discriminator_loss(self, pre, post, output, mask):
        fake = self.netD(self.discriminator_input(pre, output['pred'].detach(), mask))
        real = self.netD(self.discriminator_input(pre, post, mask))
        return 0.5 * (self.criterionGAN(fake, False) + self.criterionGAN(real, True))

    def generator_loss(self, pre, post, output, mask, q_true=None, s_true=None):
        pred = output['pred']
        domain = mask if self.args.reconstruction_domain == 'valid' else None
        adversarial = self.criterionGAN(self.netD(self.discriminator_input(pre, pred, mask)), True)
        image = masked_mean((pred - post).abs(), domain)
        gradient = self.criterionGrad(pred, post, domain)
        vector, sia = pred.new_zeros(()), pred.new_zeros(())
        if self.netSIA is not None:
            if self.args.sia_mode == 'vector':
                if q_true is None or q_true.shape != (len(pre), 2) or not torch.isfinite(q_true).all():
                    raise ValueError('Physical X/Y ground truth required for vector SIA training')
                vector = F.mse_loss(output['sia_state'], (q_true-self.sia_mean)/self.sia_std)
            if s_true is None and q_true is not None and q_true.shape == (len(pre),2):
                s_true = torch.linalg.vector_norm(q_true, dim=-1)
            if s_true is None or s_true.shape != (len(pre),) or not torch.isfinite(s_true).all() or (s_true < 0).any():
                raise ValueError('Finite nonnegative physical SIA labels [B] required')
            error = (output['sia'] - s_true)/self.args.sia_ref
            sia = error.abs().mean() if self.args.sia_loss == 'l1' else F.huber_loss(
                error, torch.zeros_like(error), delta=self.args.huber_delta)
        total = (self.args.lambda_gan * adversarial + self.args.lambda_L1 * image +
                 self.args.lambda_grad * gradient + self.args.lambda_vector * vector + self.args.lambda_sia * sia)
        return dict(loss=total, loss_G_GAN=adversarial, loss_image=image,
                    loss_gradient=gradient, loss_vector=vector, loss_sia=sia)
