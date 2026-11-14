"""Clinical channel gains shared across preoperative mean/quadrupole moments."""
import torch
from torch import nn
from model.axial_sia import axial_moments
from model.polar_sia import PolarSIAFlowMatching


class ClinicalMomentPolarSIAFlowMatching(PolarSIAFlowMatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.unet.clinical_encoder is None:
            raise ValueError('Clinical moment modulation requires clinical inputs')
        width = self.unet.clinical_encoder.feature_embedding.shape[1]
        # Identity initialization without changing the parent initialization/RNG.
        with torch.random.fork_rng(devices=[]):
            self.moment_gain = nn.Linear(width, 512)
            nn.init.zeros_(self.moment_gain.weight)
            nn.init.zeros_(self.moment_gain.bias)

    def sia_features(self, condition_features, clinical):
        moments = axial_moments(condition_features[-1]).reshape(-1, 3, 512)
        embedding, _ = self.unet.clinical_encoder(clinical)
        # Positive (0,2) gains: no direction-dependent scaling or added axis bias.
        gain = 1 + torch.tanh(self.moment_gain(embedding))
        return torch.cat(((moments * gain[:, None]).flatten(1), embedding), dim=1)
