
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

    def forward(self,pre_op,post_op):

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

        noise_pred = self.unet(x_t,t,cond_feature)

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
    
    ):

        if self.sampler == "ddpm":
            return self.ddpm_sample(pre_op)

        elif self.sampler == "ddim":
            return self.ddim_sample(
                pre_op,
                steps=self.sample_steps
            )

        else:
            raise ValueError
    # ----------------------------------
    # Reverse diffusion sampling
    # ----------------------------------

    @torch.no_grad()
    def ddpm_sample(self,pre_op):

        """
        Generate post-op image

        input:
            pre-op

        output:
            generated post-op
        """


        device = pre_op.device
        B = pre_op.shape[0]

        # condition

        cond_feature = self.condition_encoder(pre_op)
        # start from gaussian noise
        x = torch.randn_like(pre_op)
        betas = self.scheduler.betas.to(device)
        alphas = self.scheduler.alphas.to(device)
        alpha_bars = self.scheduler.alpha_bars.to(device)



        for t in reversed(range(self.timesteps)):
            timestep = torch.full((B,), t,device=device,dtype=torch.long)



            # predict noise
            noise_pred = self.unet(x,timestep,cond_feature)
            alpha = alphas[t]
            alpha_bar = alpha_bars[t]
            beta = betas[t]

            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)


            # DDPM reverse step

            x = (1 / torch.sqrt(alpha)) * (x-((1-alpha)/ torch.sqrt(1-alpha_bar))*noise_pred) + torch.sqrt(beta)*z



        return x
    

    @torch.no_grad()
    def ddim_sample(
        self,
        pre_op,
        steps=50,
        eta=0.0
    ):
        """
        DDIM Sampling

        Args:
            pre_op : [B,1,H,W]
            steps  : sampling steps
            eta    : 0 -> deterministic DDIM
                    1 -> close to DDPM
        """

        device = pre_op.device
        B = pre_op.shape[0]

        cond_feature = self.condition_encoder(pre_op)

        x = torch.randn_like(pre_op)

        alpha_bars = self.scheduler.alpha_bars.to(device)

        # timestep sequence
        seq = torch.linspace(
            self.timesteps - 1,
            0,
            steps,
            dtype=torch.long,
            device=device
        )

        for i in range(len(seq) - 1):

            t = seq[i]
            t_next = seq[i + 1]

            timestep = torch.full(
                (B,),
                t,
                device=device,
                dtype=torch.long
            )

            noise_pred = self.unet(
                x,
                timestep,
                cond_feature
            )

            alpha_bar = alpha_bars[t]
            alpha_bar_next = alpha_bars[t_next]

            # -----------------------------
            # predict x0
            # -----------------------------
            x0 = (
                x -
                torch.sqrt(1 - alpha_bar) * noise_pred
            ) / torch.sqrt(alpha_bar)

            # optional clipping
            x0 = x0.clamp(0, 1)

            # -----------------------------
            # sigma
            # -----------------------------
            sigma = (
                eta *
                torch.sqrt(
                    (1 - alpha_bar_next)
                    / (1 - alpha_bar)
                    * (1 - alpha_bar / alpha_bar_next)
                )
            )

            if eta > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            direction = torch.sqrt(
                1 - alpha_bar_next - sigma**2
            ) * noise_pred

            x = (
                torch.sqrt(alpha_bar_next) * x0
                + direction
                + sigma * noise
            )

        return x
