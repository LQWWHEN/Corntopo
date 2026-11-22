"""Split the image velocity into two preoperative directional modes and its complement."""

import torch
from torch import nn

from model.fm_common import ode_step
from model.polar_sia import PolarSIAFlowMatching, decode_polar


def optical_basis(pre_mask):
    """Return dual analysis and synthesis maps on each preoperative aperture."""
    if pre_mask is None or pre_mask.ndim != 4 or pre_mask.shape[1] != 1:
        raise ValueError("The optical modes require a single preoperative validity map per eye")
    if not pre_mask.is_floating_point():
        pre_mask = pre_mask.float()
    h, w = pre_mask.shape[-2:]
    y = 1 - 2 * (torch.arange(h, device=pre_mask.device, dtype=pre_mask.dtype) + 0.5) / h
    x = 2 * (torch.arange(w, device=pre_mask.device, dtype=pre_mask.dtype) + 0.5) / w - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r2 = xx.square() + yy.square()
    angular = torch.stack((xx.square() - yy.square(), 2 * xx * yy)) / r2.clamp_min(
        torch.finfo(x.dtype).tiny
    )
    observed = (pre_mask > 0) & (r2 > 0)
    central = observed & (r2 <= 0.25)
    mass = central.sum((-2, -1), keepdim=True).clamp_min(1)
    centered = angular - (angular * central).sum((-2, -1), keepdim=True) / mass
    design = centered * central
    a, b = design.unbind(1)
    count = mass[:, 0, 0, 0]
    g00 = a.square().sum((-2, -1)) / count
    g01 = (a * b).sum((-2, -1)) / count
    g11 = b.square().sum((-2, -1)) / count
    det = g00 * g11 - g01.square()
    trace = g00 + g11
    # Ignore both modes when the aperture cannot resolve both components.
    usable = (trace > 0) & (det >= 0.02 * trace)
    scale = usable.to(a.dtype) / (count * torch.where(usable, det, torch.ones_like(det)))
    inv = torch.stack((g11, -g01, -g01, g00), dim=1).reshape(-1, 2, 2)
    reader = torch.einsum("bij,bjhw->bihw", inv, design) * scale[:, None, None, None]
    taper = ((0.75 - r2.sqrt()) / 0.25).clamp(0, 1)
    writer = centered * taper * observed * usable[:, None, None, None]
    return reader, writer


def set_velocity_modes(velocity, desired, basis):
    reader, writer = basis
    current = (reader * velocity).sum((-2, -1))
    return velocity + ((desired - current)[:, :, None, None] * writer).sum(1, keepdim=True)


class OpticalModePolarSIA(PolarSIAFlowMatching):
    uses_preoperative_mask = True

    def __init__(self, *args, optical_image_scale, **kwargs):
        super().__init__(*args, **kwargs)
        self.optical_image_scale = optical_image_scale
        with torch.random.fork_rng(devices=[]):
            self.optical_head = nn.Linear(self.sia_readout[-1].in_features, 2, bias=False)
            self.optical_to_polar = nn.Linear(2, 2, bias=False)
        nn.init.zeros_(self.optical_head.weight)
        nn.init.zeros_(self.optical_to_polar.weight)

    def hidden_and_modes(self, condition, clinical):
        hidden = self.sia_readout[1](self.sia_readout[0](self.sia_features(condition, clinical)))
        return hidden, self.optical_head(hidden)

    def polar_outputs(self, hidden, modes):
        return decode_polar(self.sia_readout[-1](hidden) + self.optical_to_polar(modes / self.sia_ref))

    def predict_polar(self, condition, clinical):
        return self.polar_outputs(*self.hidden_and_modes(condition, clinical))

    def image_velocity(self, state, time, condition, clinical, pre_mask=None):
        _, modes = self.hidden_and_modes(condition, clinical)
        free_velocity = self.unet(state, time, condition, clinical=clinical)
        return set_velocity_modes(
            free_velocity, modes / self.optical_image_scale, optical_basis(pre_mask)
        )

    @torch.no_grad()
    def sample(self, pre, clinical=None, noise=None, steps=None, method=None, pre_mask=None):
        steps = self.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError("steps must be positive")
        condition = self.condition_encoder(pre)
        hidden, modes = self.hidden_and_modes(condition, clinical)
        q, magnitude, _ = self.polar_outputs(hidden, modes)
        desired = modes / self.optical_image_scale
        basis = optical_basis(pre_mask)
        prepared = self.unet.prepare_condition(condition, clinical)

        def field(state, time):
            t = pre.new_full((pre.shape[0],), time)
            velocity = self.unet(state[0], t, prepared, clinical=clinical)
            return (set_velocity_modes(velocity, desired, basis),)

        state = (pre,)
        for k in range(steps):
            state = ode_step(field, state, k / steps, 1 / steps, method or self.method)
        return dict(
            pred=state[0], sia_vector=q, sia=magnitude, sia_state=(q - self.sia_mean) / self.sia_std
        )
