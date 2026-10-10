"""Image-only FM baseline with the same fixed-step solver as FM-SIA."""
import torch
from torch import nn
from model.fm_common import masked_mse, ode_step


class FlowMatching(nn.Module):
    def __init__(self, condition_encoder, unet, method="euler", sample_steps=50):
        super().__init__()
        self.condition_encoder, self.unet = condition_encoder, unet
        self.method, self.sample_steps = method, sample_steps

    def forward(self, pre, post, clinical=None, mask=None):
        t = torch.rand(pre.shape[0], device=pre.device, dtype=pre.dtype)
        tau = t[:, None, None, None]
        xt = (1-tau)*pre + tau*post
        target = post-pre
        pred = self.unet(xt, t, self.condition_encoder(pre), clinical=clinical)
        loss = masked_mse(pred, target, mask)
        return dict(loss=loss, loss_image=loss, velocity_pred=pred, velocity=target, xt=xt, t=t)

    @torch.no_grad()
    def sample(self, pre, steps=None, method=None, clinical=None):
        steps = self.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError("steps must be positive")
        cond = self.condition_encoder(pre)
        def field(state, t):
            ts = pre.new_full((pre.shape[0],), t)
            return (self.unet(state[0], ts, cond, clinical=clinical),)
        state = (pre,)
        for k in range(steps):
            state = ode_step(field, state, k/steps, 1/steps, method or self.method)
        return state[0]

    def inference(self, x, clinical=None):
        return self.sample(x, clinical=clinical)
