"""Clinical modulation along the second angular harmonic of each preoperative feature."""
import torch
from torch import nn

from model.unet import ClinicalFusion


def meridional_contrast(image):
    height, width = image.shape[-2:]
    y = (torch.arange(height, device=image.device, dtype=image.dtype) + 0.5) * (2 / height) - 1
    x = (torch.arange(width, device=image.device, dtype=image.dtype) + 0.5) * (2 / width) - 1
    y = y[:, None]
    taper = (1 - x.square() - y.square()).clamp_min(0)
    mass = taper.sum().clamp_min(1)
    cosine, sine = x.square() - y.square(), 2 * x * y
    cosine = cosine - (taper * cosine).sum() / mass
    sine = sine - (taper * sine).sum() / mass
    spatial = (-2, -1)
    centered = image - (image * taper).sum(spatial, keepdim=True) / mass
    weighted = centered * taper
    c = (weighted * cosine).sum(spatial, keepdim=True) / mass
    s = (weighted * sine).sum(spatial, keepdim=True) / mass
    # Normalize by feature contrast, so weakly directional features have a weak field.
    rms = ((weighted * centered).sum(spatial, keepdim=True) / mass + 1e-6).sqrt()
    return taper * (c * cosine + s * sine) / rms


class MeridionalFiLM(ClinicalFusion):
    def __init__(self, channels, embed_dim):
        super().__init__(channels, embed_dim, 'film')
        with torch.random.fork_rng(devices=[]):
            self.direction_affine = nn.Linear(embed_dim, 2 * channels, bias=False)
        nn.init.zeros_(self.direction_affine.weight)

    def forward(self, image, global_embedding, feature_tokens):
        feature = super().forward(image, global_embedding, feature_tokens)
        gain, shift = self.direction_affine(global_embedding)[:, :, None, None].chunk(2, dim=1)
        return feature + (image * gain + shift) * meridional_contrast(image)
