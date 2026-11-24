"""Use the strength of preoperative directional moments to query clinical fields."""
import math

import torch
from torch import nn

from model.axial_sia import axial_moments
from model.polar_sia import PolarSIAFlowMatching


class ClinicalFieldPolarSIA(PolarSIAFlowMatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels = self.unet.condition_projection[-1].in_channels
        width = self.unet.clinical_encoder.feature_embedding.shape[1]
        with torch.random.fork_rng(devices=[]):
            self.field_query = nn.Linear(channels, width, bias=False)
        nn.init.zeros_(self.field_query.weight)

    def field_weights(self, directions, tokens):
        strength = torch.linalg.vector_norm(directions, dim=1)
        query = self.field_query(strength)
        scores = (query[:, None] * tokens).sum(-1) / math.sqrt(query.shape[-1])
        return scores.softmax(dim=1)

    def sia_features(self, condition_features, clinical):
        deep = condition_features[-1]
        moments = axial_moments(deep).reshape(deep.shape[0], 3, deep.shape[1])
        embedding, tokens = self.unet.clinical_encoder(clinical)
        weights = self.field_weights(moments[:, 1:], tokens)
        centered_weights = weights - 1. / tokens.shape[1]
        residual = (centered_weights[:, :, None] * tokens).sum(1)
        return torch.cat((moments.flatten(1), embedding + residual), dim=1)
