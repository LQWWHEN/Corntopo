"""Route the deepest image-conditioning gradient around the SIA readout moments."""
from functools import lru_cache

import torch

from model.axial_sia import axial_weights
from model.conditional_fno import ConditionalFNO


@lru_cache(maxsize=16)
def moment_basis(height, width):
    """Orthonormal rows spanning the actual discretized axial readout."""
    grid = torch.empty(height, width, dtype=torch.float64, device="cpu")
    weights = axial_weights(grid).flatten(1)
    _, singular_values, rows = torch.linalg.svd(weights, full_matrices=False)
    tolerance = max(weights.shape) * torch.finfo(weights.dtype).eps * singular_values.max()
    return rows[singular_values > tolerance].contiguous()


class ImageConditionGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature):
        return feature

    @staticmethod
    def backward(ctx, gradient):
        basis = moment_basis(*gradient.shape[-2:]).to(gradient)
        flat = gradient.flatten(-2)
        parallel = (flat @ basis.T) @ basis
        return (flat - parallel).reshape_as(gradient)


class MomentRoutedFNO(ConditionalFNO):
    def prepare_condition(self, condition_features, clinical=None):
        deepest = condition_features[-1]
        if self.training and deepest.requires_grad and torch.is_grad_enabled():
            condition_features = (*condition_features[:-1], ImageConditionGradient.apply(deepest))
        return super().prepare_condition(condition_features, clinical)
