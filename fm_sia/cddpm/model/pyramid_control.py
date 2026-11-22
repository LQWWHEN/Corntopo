"""Inject preoperative shape contrasts into the four hidden Fourier blocks."""
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from model.conditional_fno import ConditionalFNO
from model.polar_sia import MaskedImageConditionPolarSIA as PyramidControlledPolarSIA


class PreparedPyramid(NamedTuple):
    feature: torch.Tensor
    controls: tuple


class PyramidControlledFNO(ConditionalFNO):
    def __init__(self, *args, hidden_channels=64, **kwargs):
        super().__init__(*args, hidden_channels=hidden_channels, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.control_projections = nn.ModuleList(
                nn.Conv2d(c, hidden_channels, 1, bias=False) for c in (512, 512, 256, 128)
            )
        for projection in self.control_projections:
            nn.init.zeros_(projection.weight)

    def prepare_condition(self, condition_features, clinical=None, pre_mask=None):
        feature = super().prepare_condition(condition_features, clinical)
        if pre_mask is None or pre_mask.shape != feature[:, :1].shape:
            raise ValueError("Pyramid control needs the preoperative valid mask on the input grid")
        support = pre_mask.to(feature)
        area = support.sum((-2, -1), keepdim=True).clamp_min(1)
        padding = self.fno.domain_padding
        controls = []
        for projection, level in zip(self.control_projections, reversed(condition_features)):
            control = projection(level)
            if control.shape[-2:] != feature.shape[-2:]:
                control = F.interpolate(control, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            mean = (control * support).sum((-2, -1), keepdim=True) / area
            control = (control - mean) * support
            controls.append(padding.pad(control) if padding is not None else control)
        return PreparedPyramid(feature, tuple(controls))

    def forward(self, x, t, cond, clinical=None, pre_mask=None):
        prepared = cond if isinstance(cond, PreparedPyramid) else self.prepare_condition(cond, clinical, pre_mask)
        time = self.time_projection(self.time_emb(t))[:, :, None, None]
        x = torch.cat((x, prepared.feature + time), dim=1)
        op = self.fno
        x = op.lifting(op.positional_embedding(x))
        if op.domain_padding is not None:
            x = op.domain_padding.pad(x)
        for index, control in enumerate(prepared.controls):
            x = op.fno_blocks(x + control, index)
        if op.domain_padding is not None:
            x = op.domain_padding.unpad(x)
        return op.projection(x)
