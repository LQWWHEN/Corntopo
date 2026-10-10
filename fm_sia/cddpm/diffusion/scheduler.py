# schedule.py

import torch
import torch.nn as nn


class GaussianScheduler:
    """
    DDPM beta scheduler

    x_t = sqrt(alpha_bar)*x_0
          +
          sqrt(1-alpha_bar)*noise
    """

    def __init__(
        self,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    ):

        self.timesteps = timesteps


        self.betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps
        )


        self.alphas = 1.0 - self.betas


        self.alpha_bars = torch.cumprod(
            self.alphas,
            dim=0
        )


    def to(self,device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)

        return self


class BrownianScheduler(nn.Module):
    """
    Brownian Bridge Scheduler

    x_t = (1 - m_t) * x_A + m_t * x_B + sigma_t * eps

    x_A : source image (pre-op)
    x_B : target image (post-op)
    """

    def __init__(
        self,
        timesteps=1000,
        mt_type="linear",
        max_var=1.0
    ):
        super().__init__()
        self.timesteps = timesteps
        self.mt_type = mt_type
        self.max_var = max_var

        # -----------------------------
        # m_t
        # -----------------------------
        t = torch.linspace(0, 1, timesteps)

        if mt_type == "linear":
            m_t = t
        elif mt_type == "sin":
            m_t = torch.sin(t * torch.pi / 2)
        elif mt_type == "cos":
            m_t = 1 - torch.cos(t * torch.pi / 2)
        else:
            raise ValueError(
                f"Unknown mt_type: {mt_type}"
            )

        #self.m_t = m_t

        # -----------------------------
        # variance
        #
        # sigma^2 = max_var * m(1-m)
        # -----------------------------
        variance =2.* max_var * m_t * (1 - m_t)

        variance = torch.clamp(
            variance,
            min=1e-8
        )



  
        self.register_buffer(
            "m_t",
            m_t
        )

        self.register_buffer(
            "variance",
            variance
        )

        self.register_buffer(
            "sigma",
            torch.sqrt(variance)
        )
