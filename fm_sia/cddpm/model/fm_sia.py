"""Joint FRONT image / SIA flow: vector (2D) or direct magnitude (1D)."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.fm_common import masked_mse, ode_step


def fixed_sia_noise(labels, seed=42, sample_index=0, state_dim=2):
    """Stable per-label CPU RNG; invariant to batch order, size and device."""
    if state_dim not in (1, 2):
        raise ValueError("SIA state_dim must be 1 or 2")
    values = []
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().tolist()
    for label in labels:
        key = f"{seed}|{sample_index}|{label}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63-1)
        generator = torch.Generator(device="cpu").manual_seed(value)
        values.append(torch.randn(state_dim, generator=generator))
    return torch.stack(values)


class JointFlowMatching(nn.Module):
    def __init__(self, condition_encoder, unet, sia_mean, sia_std, sample_steps=100,
                 method="euler", lambda_vector=1.0, lambda_sia=0.1,
                 sia_ref=1.0, huber_delta=1.0, sia_mode="vector", lambda_scalar=1.0):
        super().__init__()
        self.condition_encoder, self.unet = condition_encoder, unet
        if sia_mode not in ('vector', 'scalar'):
            raise ValueError("sia_mode must be vector or scalar")
        self.sia_mode = sia_mode
        self.state_dim = 2 if sia_mode == 'vector' else 1
        mean, std = torch.as_tensor(sia_mean, dtype=torch.float32), torch.as_tensor(sia_std, dtype=torch.float32)
        if mean.shape != (self.state_dim,) or std.shape != (self.state_dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"SIA normalization requires finite length-{self.state_dim} mean and positive std")
        active_weight = lambda_vector if sia_mode == 'vector' else lambda_scalar
        if not all(math.isfinite(v) for v in (active_weight, lambda_sia, sia_ref, huber_delta)) or sample_steps < 1 or active_weight <= 0 or lambda_sia < 0 or sia_ref <= 0 or huber_delta <= 0:
            raise ValueError("Invalid FM-SIA loss or sampler settings")
        self.register_buffer("sia_mean", mean)
        self.register_buffer("sia_std", std)
        self.sample_steps, self.method = sample_steps, method
        self.lambda_vector, self.lambda_sia = lambda_vector, lambda_sia
        self.lambda_scalar = lambda_scalar
        self.sia_ref, self.huber_delta = sia_ref, huber_delta

    def decode(self, z):
        q = self.sia_mean + self.sia_std*z
        if self.sia_mode == 'scalar':
            # Training uses this unprojected endpoint to retain gradients below 0 D.
            return None, q.squeeze(-1)
        return q, torch.linalg.vector_norm(q, dim=-1)

    def forward(self, pre, post, sia_vector=None, clinical=None, mask=None, t=None, noise=None,
                sia_magnitude=None):
        batch = pre.shape[0]
        if self.sia_mode == 'vector':
            if sia_vector is None or sia_vector.shape != (batch, 2) or not torch.isfinite(sia_vector).all():
                raise ValueError("sia_vector must be finite [B,2] in physical X/Y units (D)")
            physical = sia_vector.to(pre)
            s_true = torch.linalg.vector_norm(physical, dim=-1)
        else:
            if sia_magnitude is None or sia_magnitude.shape not in ((batch,), (batch, 1)) or not torch.isfinite(sia_magnitude).all() or (sia_magnitude < 0).any():
                raise ValueError("sia_magnitude must be finite nonnegative [B] or [B,1] in D")
            physical = sia_magnitude.to(pre).reshape(batch, 1)
            s_true = physical[:, 0]
        t = torch.rand(batch, device=pre.device, dtype=pre.dtype) if t is None else t.to(pre)
        noise = torch.randn_like(physical) if noise is None else noise.to(pre)
        if t.shape != (batch,) or noise.shape != (batch, self.state_dim) or not torch.isfinite(t).all() or not torch.isfinite(noise).all() or ((t < 0) | (t > 1)).any():
            raise ValueError("Invalid time or noise shape")
        z1 = (physical-self.sia_mean)/self.sia_std
        xt = (1-t[:, None, None, None])*pre + t[:, None, None, None]*post
        zt = (1-t[:, None])*noise + t[:, None]*z1
        vx, vz = self.unet(xt, t, self.condition_encoder(pre), clinical=clinical, sia_state=zt)
        loss_image = masked_mse(vx, post-pre, mask)
        loss_state = F.mse_loss(vz, z1-noise)
        # One-step endpoint estimate; final evaluation always integrates the ODE.
        z_clean = zt + (1-t[:, None])*vz
        q_clean, s_clean = self.decode(z_clean)
        loss_sia = F.huber_loss((s_clean-s_true)/self.sia_ref,
                               torch.zeros_like(s_true), delta=self.huber_delta)
        weight = self.lambda_vector if self.sia_mode == 'vector' else self.lambda_scalar
        loss = loss_image + weight*loss_state + self.lambda_sia*loss_sia
        result = dict(loss=loss, loss_image=loss_image, loss_sia=loss_sia,
                      velocity_pred=vx, sia_velocity_pred=vz, sia_clean=s_clean, t=t)
        result['loss_vector' if self.sia_mode == 'vector' else 'loss_scalar'] = loss_state
        if self.sia_mode == 'vector':
            result['sia_vector_clean'] = q_clean
        return result

    @torch.no_grad()
    def sample(self, pre, clinical=None, noise=None, steps=None, method=None):
        steps = self.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError("steps must be positive")
        z0 = torch.randn(pre.shape[0], self.state_dim, device=pre.device, dtype=pre.dtype) if noise is None else noise.to(pre)
        if z0.shape != (pre.shape[0], self.state_dim) or not torch.isfinite(z0).all():
            raise ValueError(f"Initial noise must be finite [B,{self.state_dim}]")
        cond = self.condition_encoder(pre)
        def field(state, time):
            t = pre.new_full((pre.shape[0],), time)
            return self.unet(state[0], t, cond, clinical=clinical, sia_state=state[1])
        state = (pre, z0)
        for k in range(steps):
            state = ode_step(field, state, k/steps, 1/steps, method or self.method)
        q, s = self.decode(state[1])
        result = dict(pred=state[0], sia=s, sia_state=state[1])
        if self.sia_mode == 'vector':
            result['sia_vector'] = q
        else:
            # Physical output is nonnegative; keep raw values for diagnostics.
            result.update(sia=s.clamp_min(0), sia_unclipped=s)
        return result

    def inference(self, x, clinical=None, noise=None):
        return self.sample(x, clinical=clinical, noise=noise)

