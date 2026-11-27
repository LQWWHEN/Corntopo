"""Preoperative coverage-weighted, centered spatial moments for polar SIA."""
import torch
from torch.nn import functional as F
from model.polar_sia import PolarSIAFlowMatching, decode_polar


def aperture_moments(feature, pre_mask, *, center_basis=True, half_turn_average=False):
    """Pool features over the measured disk, optionally removing the constant mode."""
    if pre_mask is None or pre_mask.ndim != 4 or pre_mask.shape[:2] != (feature.shape[0], 1):
        raise ValueError("A preoperative valid mask of shape [B,1,H,W] is required")
    h, w = feature.shape[-2:]
    yy, xx = torch.meshgrid(
        1 - 2 * (torch.arange(h, device=feature.device, dtype=feature.dtype) + .5) / h,
        2 * (torch.arange(w, device=feature.device, dtype=feature.dtype) + .5) / w - 1,
        indexing="ij")
    # Area pooling retains the measured fraction of each coarse feature cell.
    coverage = F.interpolate(pre_mask.to(feature), size=(h, w), mode="area")
    support = coverage * (xx.square() + yy.square() <= 1)
    total = support.sum((-2, -1), keepdim=True)
    mean_weights = support / total.clamp_min(torch.finfo(feature.dtype).eps)
    basis = torch.stack((xx.square() - yy.square(), 2 * xx * yy))
    if center_basis:
        basis = basis - (mean_weights * basis).sum((-2, -1), keepdim=True)
    signed = support * basis
    # Match the parent on a full symmetric disk. Tiny supports cannot amplify
    # one boundary cell beyond the parent's minimum normalization mass.
    signed = signed / signed.abs().sum((-2, -1), keepdim=True).clamp_min(1)
    weights = torch.cat((mean_weights, signed), dim=1)
    if half_turn_average:
        # Remove only the odd part of each patient's three pooling kernels.
        weights = .5 * (weights + weights.flip((-2, -1)))
    return torch.einsum("bchw,bkhw->bkc", feature, weights).flatten(1)


class AperturePolarSIAFlowMatching(PolarSIAFlowMatching):
    uses_preoperative_mask = True

    def __init__(self, *args, center_basis=True, half_turn_average=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_basis = center_basis
        self.half_turn_average = half_turn_average

    def sia_mask_argument(self, pre_mask):
        return {"pre_mask": pre_mask}

    def predict_polar(self, condition, clinical, *, pre_mask):
        pooled = aperture_moments(condition[-1], pre_mask, center_basis=self.center_basis,
                                  half_turn_average=self.half_turn_average)
        if self.unet.clinical_encoder is not None:
            embedding, _ = self.unet.clinical_encoder(clinical)
            pooled = torch.cat((pooled, embedding), dim=1)
        return decode_polar(self.sia_readout(pooled))

    def predict_sia(self, condition, clinical, *, pre_mask):
        return self.predict_polar(condition, clinical, pre_mask=pre_mask)[0]
