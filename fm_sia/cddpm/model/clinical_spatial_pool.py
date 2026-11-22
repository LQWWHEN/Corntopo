"""Clinical-query spatial pooling of the zeroth-order preoperative SIA feature."""
import torch
from torch import nn

from model.axial_sia import axial_weights
from model.polar_sia import PolarSIAFlowMatching


class ClinicalSpatialPool(nn.Module):
    def __init__(self, channels, clinical_width, key_width=16):
        super().__init__()
        self.key = nn.Conv2d(channels, key_width, 1, bias=False)
        self.query = nn.Linear(clinical_width, key_width, bias=False)
        nn.init.zeros_(self.query.weight)

    def forward(self, feature, clinical_embedding):
        weights = axial_weights(feature)
        moments = torch.einsum("bchw,khw->bkc", feature, weights)
        support = weights[0].flatten() > 0
        keys = self.key(feature).flatten(2)
        query = self.query(clinical_embedding)
        logits = torch.einsum("bc,bcn->bn", query, keys) * (keys.shape[1] ** -0.5)
        attention = logits.masked_fill(~support, -torch.inf).softmax(-1)
        uniform = torch.zeros_like(logits).masked_fill(~support, -torch.inf).softmax(-1)
        shift = torch.einsum("bn,bcn->bc", attention - uniform, feature.flatten(2))
        return torch.cat((moments[:, 0] + shift, moments[:, 1:].flatten(1)), dim=1)


class ClinicalSpatialPolarSIA(PolarSIAFlowMatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        clinical_width = self.unet.clinical_encoder.feature_embedding.shape[1]
        with torch.random.fork_rng(devices=[]):
            self.spatial_pool = ClinicalSpatialPool(512, clinical_width)

    def sia_features(self, condition_features, clinical):
        embedding, _ = self.unet.clinical_encoder(clinical)
        moments = self.spatial_pool(condition_features[-1], embedding)
        return torch.cat((moments, embedding), dim=1)
