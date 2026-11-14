"""Signed, native-pixel power-change gradients on pairs of valid pixels."""
import torch


def masked_gradient_mse(pred, target, mask=None):
    """Equal mean over cases of squared error on horizontal/vertical edges.

    A valid edge has two valid endpoints. There is no padding or wrapping,
    and no division by physical distance: the experiment fixes128x128 pixels.
    Cases without any eligible edge contribute zero derivative supervision.
    """
    error = pred-target
    valid = torch.ones_like(error) if mask is None else mask.to(error).expand_as(error)
    horizontal = valid[..., 1:]*valid[..., :-1]
    vertical = valid[..., 1:, :]*valid[..., :-1, :]
    dx = error[..., 1:]-error[..., :-1]
    dy = error[..., 1:, :]-error[..., :-1, :]
    total = (dx.square()*horizontal).flatten(1).sum(1)+(dy.square()*vertical).flatten(1).sum(1)
    count = horizontal.flatten(1).sum(1)+vertical.flatten(1).sum(1)
    return (total/count.clamp_min(1)).mean()
