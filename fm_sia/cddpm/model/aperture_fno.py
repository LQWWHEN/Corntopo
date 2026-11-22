"""Restrict each Fourier velocity block to the observed preoperative aperture."""
from typing import NamedTuple

import torch

from model.conditional_fno import ConditionalFNO


class ApertureCondition(NamedTuple):
    feature: torch.Tensor
    support: torch.Tensor
    padded_support: torch.Tensor


class ApertureFNO(ConditionalFNO):
    def prepare_condition(self, condition_features, clinical=None, pre_mask=None):
        feature = super().prepare_condition(condition_features, clinical)
        if pre_mask is None or pre_mask.shape != feature[:, :1].shape:
            raise ValueError("Aperture FNO needs one preoperative mask per image on the input grid")
        support = (pre_mask > 0).to(feature)
        padding = self.fno.domain_padding
        padded_support = padding.pad(support) if padding is not None else support
        return ApertureCondition(feature, support, padded_support)

    def forward(self, x, t, cond, clinical=None, pre_mask=None):
        prepared = cond if isinstance(cond, ApertureCondition) else self.prepare_condition(cond, clinical, pre_mask)
        time = self.time_projection(self.time_emb(t))[:, :, None, None]
        state = torch.cat((x, prepared.feature + time), dim=1)
        op = self.fno
        hidden = op.lifting(op.positional_embedding(state))
        if op.domain_padding is not None:
            hidden = op.domain_padding.pad(hidden)
        for index in range(op.n_layers):
            hidden = op.fno_blocks(hidden * prepared.padded_support, index)
        if op.domain_padding is not None:
            hidden = op.domain_padding.unpad(hidden)
        return op.projection(hidden) * prepared.support
