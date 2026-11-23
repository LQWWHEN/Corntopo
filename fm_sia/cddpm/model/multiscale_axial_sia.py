"""Share the image-condition channel projections with a multiscale axial SIA readout."""
import torch
from torch import nn
from torch.nn import functional as F

from model.axial_sia import axial_moments
from model.polar_sia import PolarSIAFlowMatching


class MultiscaleAxialPolarSIA(PolarSIAFlowMatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels = sum(layer.out_channels for layer in self.unet.condition_projection)
        deep_channels = self.unet.condition_projection[-1].in_channels
        with torch.random.fork_rng(devices=[]):
            self.axial_scale_projection = nn.Linear(channels, deep_channels, bias=False)
        nn.init.zeros_(self.axial_scale_projection.weight)

    def sia_features(self, condition_features, clinical):
        moments = [axial_moments(feature).reshape(feature.shape[0], 3, feature.shape[1])
                   for feature in condition_features]
        # Pool on each native grid first. Only the channel mixing is shared with the image branch.
        directions = [F.linear(m[:, 1:], layer.weight[:, :, 0, 0])
                      for m, layer in zip(moments, self.unet.condition_projection)]
        correction = self.axial_scale_projection(torch.cat(directions, dim=-1))
        deep = moments[-1]
        embedding, _ = self.unet.clinical_encoder(clinical)
        return torch.cat((deep[:, 0], (deep[:, 1:] + correction).flatten(1), embedding), dim=1)
