"""Losses with explicit pixel support and no invalid-boundary gradients."""
import torch
from torch import nn


def masked_mean(value, mask=None):
    if mask is None:
        return value.mean()
    mask = mask.to(value)
    if mask.shape != value.shape:
        raise ValueError('Mask and loss shapes differ')
    # Each eye has equal reconstruction-loss weight; metrics separately pool pixels.
    dims = tuple(range(1, value.ndim))
    count = mask.sum(dims)
    return ((value * mask).sum(dims) / count.clamp_min(1)).mean()


class GradientLoss(nn.Module):
    def forward(self, pred, target, mask=None):
        error = pred - target
        dx = (error[..., 1:] - error[..., :-1]).abs()
        dy = (error[..., 1:, :] - error[..., :-1, :]).abs()
        mx = None if mask is None else mask[..., 1:] * mask[..., :-1]
        my = None if mask is None else mask[..., 1:, :] * mask[..., :-1, :]
        return masked_mean(dx, mx) + masked_mean(dy, my)
