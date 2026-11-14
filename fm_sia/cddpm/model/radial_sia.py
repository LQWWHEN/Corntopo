"""Central/peripheral contrasts in the preoperative polar SIA readout."""
import torch
from torch import nn
from model.polar_sia import PolarSIAFlowMatching, decode_polar


def radial_contrast_moments(feature):
    """Inner-minus-outer mean and quadrupole moments on a feature-space disk.

    The fixed equal-area partition is r^2=0.5. Coordinates and independently
    normalized signed weights match the historical global axial readout;
    learned feature moments are not calibrated corneal power measurements.
    """
    h, w = feature.shape[-2:]
    yy, xx = torch.meshgrid(
        1 - 2 * (torch.arange(h, device=feature.device, dtype=feature.dtype) + .5) / h,
        2 * (torch.arange(w, device=feature.device, dtype=feature.dtype) + .5) / w - 1,
        indexing='ij')
    radius2 = xx.square() + yy.square()
    basis = torch.stack((torch.ones_like(xx), xx.square() - yy.square(), 2 * xx * yy))
    regions = torch.stack((radius2 <= .5, (radius2 > .5) & (radius2 <= 1)))
    weights = regions[:, None].to(feature.dtype) * basis[None]
    weights = weights / weights.abs().sum((-2, -1), keepdim=True).clamp_min(1)
    contrast = weights[0] - weights[1]
    return torch.einsum('bchw,khw->bkc', feature, contrast).flatten(1)


class RadialContrastPolarSIAFlowMatching(PolarSIAFlowMatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve parent RNG and its exact original first-linear computation.
        with torch.random.fork_rng(devices=[]):
            self.radial_projection = nn.Linear(3 * 512, 128, bias=False)
            nn.init.zeros_(self.radial_projection.weight)

    def predict_polar(self, condition_features, clinical):
        parent = self.sia_readout[0](self.sia_features(condition_features, clinical))
        radial = self.radial_projection(radial_contrast_moments(condition_features[-1]))
        hidden = self.sia_readout[1](parent + radial)
        return decode_polar(self.sia_readout[2](hidden))
