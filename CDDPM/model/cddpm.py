
# cddpm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.scheduler import GaussianScheduler



class CDDPM(nn.Module):

    def __init__(
        self,
        condition_encoder,
        unet,
        timesteps=1000,
        sampler = "ddpm",
        sample_steps = 50
    ):

        super().__init__()
        self.condition_encoder = condition_encoder
        self.unet = unet
        self.timesteps = timesteps
        self.scheduler = GaussianScheduler(
            timesteps
        )
        self.sampler=  sampler
        self.sample_steps = sample_steps

    # ----------------------------------
    # Forward diffusion
    # ----------------------------------

    def q_sample(
        self,
        x0,
        t,
        noise=None
    ):

        """
        x0:
            real post-op image

        t:
            diffusion timestep

        output:
            noisy image x_t
        """


        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar = self.scheduler.alpha_bars.to(x0.device)
        sqrt_alpha_bar = torch.sqrt(alpha_bar[t])
        sqrt_one_minus = torch.sqrt(1-alpha_bar[t])

        # reshape

        sqrt_alpha_bar = sqrt_alpha_bar[:,None,None,None]
        sqrt_one_minus = sqrt_one_minus[:,None,None,None]

        xt = (sqrt_alpha_bar*x0+sqrt_one_minus*noise)

        return xt, noise



    # ----------------------------------
    # Training forward
    # ----------------------------------

    def forward(self,pre_op,post_op,clinical=None):

        """
        pre_op:
            B,1,128,128

        post_op:
            B,1,128,128
        """


        B = post_op.shape[0]
        device = post_op.device

        # random timestep
        t = torch.randint(0,self.timesteps,(B,),device=device)

        # q(x_t|x_0)
        x_t, noise = self.q_sample(post_op,t)



        # condition feature

        cond_feature = self.condition_encoder(pre_op)



        # predict noise

        noise_pred = self.unet(x_t,t,cond_feature,clinical=clinical)

        # DDPM loss

        loss = F.mse_loss(noise_pred,noise)
        return {
            "loss": loss,
            "noise_pred": noise_pred,
            "noise": noise,
            "t": t
        }
    def sample(
        self,
        pre_op,
        clinical=None,
    
    ):

        if self.sampler == "ddpm":
            return self.ddpm_sample(pre_op, clinical=clinical)
