"""Task-specific magnitude/phase hidden layers with matched initialization."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from model.polar_sia import PolarSIAFlowMatching


class ScalarReadout(nn.Module):
    def __init__(self, parent, row):
        super().__init__()
        self.hidden = deepcopy(parent[0])
        self.activation = deepcopy(parent[1])
        self.weight = nn.Parameter(parent[2].weight[row:row+1].detach().clone())
        self.bias = nn.Parameter(parent[2].bias[row:row+1].detach().clone())

    def forward(self, features):
        return F.linear(self.activation(self.hidden(features)), self.weight, self.bias)


class SplitPolarReadout(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.magnitude = ScalarReadout(parent, 0)
        self.phase = ScalarReadout(parent, 1)

    def forward(self, features):
        return torch.cat((self.magnitude(features), self.phase(features)), dim=1)


class SplitPolarSIAFlowMatching(PolarSIAFlowMatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rank_weight or self.axis_gradient_projection:
            raise ValueError('Split polar readout uses the original polar objectives')
        # Copy initialized012 parameters without consuming additional random draws.
        # The two hidden layers subsequently receive their own task gradients.
        self.sia_readout = SplitPolarReadout(self.sia_readout)
