"""A rank-contrast auxiliary objective with the physical double-angle SIA metric."""
import torch
from torch.nn import functional as F


def axial_rank_contrast(features, target, temperature=.2):
    """Rank other samples by squared physical-vector distance, preserving axis periodicity.

    The per-anchor denominator includes the candidate and every at-least-as-distant
    target. Equal distances share a rank within 1e-6 in squared D / (1 D)^2 units.
    This is a single-view auxiliary adaptation of Rank-N-Contrast (Zha et al., 2023).
    """
    batch = features.shape[0]
    if batch < 2:
        return features.sum()*0
    unit = F.normalize(features, dim=-1)
    scores = unit @ unit.T / temperature
    q = target.detach().double()
    distance = (q[:, None]-q[None]).square().sum(-1)
    other = ~torch.eye(batch, dtype=torch.bool, device=features.device)
    scores = scores[other].reshape(batch, batch-1)
    distance = distance[other].reshape(batch, batch-1)
    eligible = distance[:, None, :] >= distance[:, :, None]-1e-6
    denominator = torch.logsumexp(scores[:, None, :].expand(-1, batch-1, -1).masked_fill(~eligible, -torch.inf), dim=-1)
    return (denominator-scores).mean()
