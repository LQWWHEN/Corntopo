"""Mix channels in the tensor-product cosine basis of the padded image grid."""
import math

import torch
from torch import nn

from model.conditional_fno import ConditionalFNO


def cosine_basis(length, modes, like):
    """Return the first rows of the orthonormal type-II DCT matrix."""
    position = torch.arange(length, dtype=like.dtype, device=like.device) + 0.5
    freq = torch.arange(min(length, modes), dtype=like.dtype, device=like.device)
    basis = torch.cos(freq[:, None] * (math.pi / length) * position)
    scale = torch.full_like(freq, math.sqrt(2. / length))
    scale[0] = 1. / math.sqrt(length)
    return basis * scale[:, None]


class CosineSpectralConv(nn.Module):
    def __init__(self, channels, modes, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, channels, *modes) / math.sqrt(channels))
        self.bias = bias

    def transform(self, x, output_shape=None):
        if output_shape is not None and tuple(output_shape) != x.shape[-2:]:
            raise ValueError("Cosine spectral blocks require a fixed grid")
        return x

    def forward(self, x, output_shape=None):
        self.transform(x, output_shape)
        rows = cosine_basis(x.size(-2), self.weight.size(-2), x)
        cols = cosine_basis(x.size(-1), self.weight.size(-1), x)
        coeff = (rows @ x) @ cols.T
        weight = self.weight[:, :, :rows.size(0), :cols.size(0)]
        mixed = torch.einsum("bihw,iohw->bohw", coeff, weight)
        return (rows.T @ mixed) @ cols + self.bias


class CosineVelocityOperator(ConditionalFNO):
    def __init__(self, *, hidden_channels, modes, **kwargs):
        super().__init__(hidden_channels=hidden_channels, modes=modes, **kwargs)
        convs = self.fno.fno_blocks.convs
        with torch.random.fork_rng(devices=[]):
            for i, original in enumerate(convs):
                convs[i] = CosineSpectralConv(hidden_channels, modes, original.bias)
