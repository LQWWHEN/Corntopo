"""Four-stage neural operator with deterministic two-dimensional grid transfers."""
from functools import partial

import torch
from torch.nn import functional as F
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models import UNO


class BilinearSkipSpectralConv(SpectralConv):
    def transform(self, x, output_shape=None):
        """Transfer each real-space residual to the spectral layer's output grid."""
        shape = x.shape[-2:]
        if output_shape is None and self.resolution_scaling_factor is not None:
            output_shape = tuple(round(n * scale) for n, scale in zip(shape, self.resolution_scaling_factor))
        if output_shape is None or tuple(output_shape) == shape:
            return x
        return F.interpolate(x, size=output_shape, mode="bilinear", align_corners=True)


class UVelocityOperator(UNO):
    def __init__(self, modes, hidden_channels, domain_padding):
        coarse_modes = tuple(max(1, n // 2) for n in modes)
        super().__init__(
            in_channels=65, out_channels=1, hidden_channels=hidden_channels,
            lifting_channels=2 * hidden_channels, projection_channels=2 * hidden_channels,
            n_layers=4, uno_out_channels=[hidden_channels] * 4,
            uno_n_modes=[tuple(modes), coarse_modes, coarse_modes, tuple(modes)],
            uno_scalings=[[1, 1], [.5, .5], [2, 2], [1, 1]],
            horizontal_skips_map={3: 0}, channel_mlp_skip="linear",
            domain_padding=domain_padding, domain_padding_mode="symmetric",
            operator_block=partial(FNOBlocks, conv_module=BilinearSkipSpectralConv),
        )

    def forward(self, x):
        x = self.lifting(self.positional_embedding(x))
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        full_grid = x.shape[-2:]
        coarse_grid = tuple(max(1, round(n / 2)) for n in full_grid)
        fine = self.fno_blocks[0](x, output_shape=full_grid)
        coarse = self.fno_blocks[1](fine, output_shape=coarse_grid)
        recovered = self.fno_blocks[2](coarse, output_shape=full_grid)
        x = torch.cat((recovered, self.horizontal_skips["0"](fine)), dim=1)
        x = self.fno_blocks[3](x, output_shape=full_grid)
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        return self.projection(x)
