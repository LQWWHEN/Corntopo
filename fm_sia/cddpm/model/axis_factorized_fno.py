"""Fixed-grid meridional factorization of the conditional Fourier velocity."""
import math

import torch
from torch import nn

from model.conditional_fno import ConditionalFNO


class AxisSpectralConv(nn.Module):
    """Sum learned row and column convolutions while preserving the spatial grid."""

    def __init__(self, channels, modes, bias):
        super().__init__()
        scale = 1 / math.sqrt(2 * channels)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(channels, channels, n // 2 + 1, dtype=torch.cfloat) * scale)
            for n in modes
        ])
        self.bias = bias

    def transform(self, x, output_shape=None):
        if output_shape is not None and tuple(output_shape) != x.shape[-2:]:
            raise ValueError("Axis-factorized Fourier blocks require a fixed grid")
        return x

    def forward(self, x, output_shape=None):
        self.transform(x, output_shape)
        result = self.bias
        for axis, weight in zip((-2, -1), self.weights):
            lines = x.movedim(axis, -1).contiguous()
            spectrum = torch.fft.rfft(lines, norm="forward")
            count = min(spectrum.shape[-1], weight.shape[-1])
            mixed = torch.einsum("bilm,iom->bolm", spectrum[..., :count], weight[..., :count])
            # irfft pads the unrepresented high frequencies and restores odd lengths.
            spatial = torch.fft.irfft(mixed, n=lines.size(-1), norm="forward")
            result = result + spatial.movedim(-1, axis)
        return result


class AxisFactorizedFNO(ConditionalFNO):
    def __init__(self, *, hidden_channels, modes, **kwargs):
        super().__init__(hidden_channels=hidden_channels, modes=modes, **kwargs)
        convs = self.fno.fno_blocks.convs
        with torch.random.fork_rng(devices=[]):
            for i, original in enumerate(convs):
                convs[i] = AxisSpectralConv(hidden_channels, modes, original.bias)
