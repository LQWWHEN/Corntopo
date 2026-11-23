"""Modulate the half-turn-odd part of each preoperative encoder scale."""
import torch
from torch import nn
from model.unet import ClinicalFusion


class ParityFiLM(ClinicalFusion):
    def __init__(self, channels, embed_dim):
        super().__init__(channels, embed_dim, 'film')
        with torch.random.fork_rng(devices=[]):
            self.odd_gain = nn.Linear(embed_dim, 4, bias=False)
        nn.init.zeros_(self.odd_gain.weight)

    def forward(self, image, global_embedding, feature_tokens):
        base = super().forward(image, global_embedding, feature_tokens)
        scales = base.unflatten(1, (4, -1))
        odd = (scales - scales.flip((-2, -1))) * 0.5
        gain = self.odd_gain(global_embedding).tanh()[:, :, None, None, None]
        return (scales + gain * odd).flatten(1, 2)
