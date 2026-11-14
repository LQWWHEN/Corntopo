"""Magnitude-prioritized axis gradient routing at the shared SIA hidden state."""
import torch


def project_axis_gradient(radial, angular):
    """Per-case projection; radial is unchanged and non-conflicting rows pass through."""
    dot = (radial * angular).sum(-1, keepdim=True)
    norm2 = radial.square().sum(-1, keepdim=True)
    coefficient = dot.clamp_max(0) / norm2.clamp_min(torch.finfo(radial.dtype).tiny)
    return angular - coefficient * radial, dot.squeeze(-1) < 0


def axis_gradient_surrogate(hidden, magnitude_loss, axis_loss):
    """Zero-valued surrogate changes upstream hidden gradients, not output-head gradients."""
    radial, = torch.autograd.grad(magnitude_loss, hidden, retain_graph=True)
    angular, = torch.autograd.grad(axis_loss, hidden, retain_graph=True)
    projected, conflict = project_axis_gradient(radial, angular)
    correction = (projected - angular).detach()
    surrogate = (correction * (hidden - hidden.detach())).sum()
    return surrogate, dict(
        gradient_axis_conflict_fraction=conflict.to(hidden.dtype).mean().detach(),
        gradient_axis_dot_before=(radial * angular).sum(-1).mean().detach(),
        gradient_axis_dot_after=(radial * projected).sum(-1).mean().detach())
