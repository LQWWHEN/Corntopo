"""Input-dependent von Mises concentration for the doubled SIA phase."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AxisConcentration(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, width))
        self.bias = nn.Parameter(torch.tensor([1. + math.log(-math.expm1(-1.))]))

    def forward(self, hidden):
        return F.softplus(F.linear(hidden, self.weight, self.bias)).squeeze(-1)


def concentration_axis_loss(phase, target, concentration, minimum_magnitude):
    magnitude = target.norm(dim=-1)
    valid = (magnitude >= minimum_magnitude) & (magnitude > 1e-8)
    if not valid.any():
        return (phase.sum()+concentration.sum())*0
    error = phase[valid]-torch.atan2(target[valid, 1], target[valid, 0])
    kappa = concentration[valid]
    # i0e avoids overflow. This is von Mises NLL up to a fixed constant;
    # at kappa=1 the value and phase gradient match the012 cosine objective.
    reference = torch.special.i0e(kappa.new_ones(())).log()
    return (kappa*(1-error.cos())+torch.special.i0e(kappa).log()-reference).mean()
