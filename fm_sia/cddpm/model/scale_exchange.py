"""Patient-conditioned exchange of contrasts between four encoder scales."""
import torch
from torch import nn
from model.unet import ClinicalFusion


class ScaleExchangeFiLM(ClinicalFusion):
    def __init__(self, channels, embed_dim):
        super().__init__(channels, embed_dim, 'film')
        # Four destinations, with three distinct donors for each.
        with torch.random.fork_rng(devices=[]):
            self.exchange = nn.Linear(embed_dim, 12, bias=False)
        nn.init.zeros_(self.exchange.weight)

    def forward(self, image, global_embedding, feature_tokens):
        base = super().forward(image, global_embedding, feature_tokens)
        scales = image.unflatten(1, (4, -1))
        weights = self.exchange(global_embedding).unflatten(1, (4, 3))
        update = torch.zeros_like(scales)
        for offset in range(1, 4):
            contrast = scales.roll(-offset, dims=1) - scales
            update = update + weights[:, :, offset - 1, None, None, None] * contrast
        return base + update.flatten(1, 2)
