"""Valid-pixel low-order angular projection of a corneal FM velocity error.

Four radial bands retain local radial changes. Complete sine/cosine pairs
describe the angular mean, asymmetry and second-order astigmatic pattern.
These are image-shape descriptors, not a conversion to clinical SIA labels.
"""
import math
import torch


def annular_harmonic_loss(prediction, target, mask=None):
    """Mean energy of the signed residual projected onto angular orders 0–2.

    Coordinates use pixel centres and raw-map rows increasing upwards. Only
    common valid pixels within the inscribed disk enter the auxiliary loss;
    the caller retains its ordinary pixel loss on the full valid domain.
    Masked least squares avoids assuming complete circular sampling. The
    diagonal stabilizer handles empty or sparsely populated rings without
    selecting or discarding patients.
    """
    error = prediction-target
    if error.ndim != 4 or target.shape != prediction.shape:
        raise ValueError('Expected matching [B,C,H,W] images')
    mask = torch.ones_like(error) if mask is None else mask.to(error)
    if mask.shape != error.shape:
        raise ValueError('Mask must match images')
    h, w = error.shape[-2:]
    yy, xx = torch.meshgrid(
        2*(torch.arange(h, device=error.device, dtype=error.dtype)+.5)/h-1,
        2*(torch.arange(w, device=error.device, dtype=error.dtype)+.5)/w-1, indexing='ij')
    radius = torch.sqrt(xx.square()+yy.square())
    angle = torch.atan2(yy, xx)
    noncentral = (radius > 0).to(error)
    basis = torch.stack((torch.ones_like(angle),
                         math.sqrt(2)*angle.cos()*noncentral,
                         math.sqrt(2)*angle.sin()*noncentral,
                         math.sqrt(2)*(2*angle).cos()*noncentral,
                         math.sqrt(2)*(2*angle).sin()*noncentral)).flatten(1)
    rings = torch.stack([(radius >= i/4) &
                         (radius < (i+1)/4 if i < 3 else radius <= 1) for i in range(4)])
    weight = mask.flatten(2).unsqueeze(2)*rings.flatten(1).to(error)
    count = weight.sum(-1)
    denominator = count.clamp_min(1)
    gram = torch.einsum('bcap,kp,lp->bcakl', weight, basis, basis)/denominator[..., None, None]
    rhs = torch.einsum('bcap,kp,bcp->bcak', weight, basis, error.flatten(2))/denominator[..., None]
    beta = torch.linalg.solve(
        gram+1e-6*torch.eye(5, device=error.device, dtype=error.dtype), rhs.unsqueeze(-1)).squeeze(-1)
    energy = torch.einsum('bcak,bcakl,bcal->bca', beta, gram, beta)
    return (energy*count).sum()/count.sum().clamp_min(1)
