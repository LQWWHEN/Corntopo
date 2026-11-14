"""Factor a physical double-angle SIA vector into magnitude and periodic phase."""
import math
import torch
from torch.nn import functional as F
from model.axial_sia import AxialSIAFlowMatching
from model.fm_common import masked_mse


def decode_polar(raw):
    magnitude = F.softplus(raw[:, 0])
    phase = raw[:, 1]  # Twice the physical astigmatic axis, in radians.
    vector = magnitude[:, None] * torch.stack((phase.cos(), phase.sin()), dim=-1)
    return vector, magnitude, phase


def periodic_axis_loss(phase, target, minimum_magnitude):
    magnitude = target.norm(dim=-1)
    valid = (magnitude >= minimum_magnitude) & (magnitude > 1e-8)
    target_phase = torch.atan2(target[:, 1], target[:, 0])
    error = 1 - torch.cos(phase - target_phase)
    return error[valid].mean() if valid.any() else phase.sum()*0


class PolarSIAFlowMatching(AxialSIAFlowMatching):
    def __init__(self, *args, initial_magnitude, initial_phase, lambda_axis=1.,
                 axis_min_magnitude=.1, rank_weight=0., rank_temperature=.2,
                 image_gradient_weight=0., **kwargs):
        super().__init__(*args, **kwargs)
        if (not all(math.isfinite(v) for v in (initial_magnitude, initial_phase, lambda_axis, axis_min_magnitude))
                or initial_magnitude < 0 or lambda_axis <= 0 or axis_min_magnitude < 0):
            raise ValueError('Invalid polar initialization or axis objective')
        self.lambda_axis = lambda_axis
        self.axis_min_magnitude = axis_min_magnitude
        if not math.isfinite(image_gradient_weight) or image_gradient_weight < 0:
            raise ValueError('Image gradient weight must be finite and nonnegative')
        self.image_gradient_weight = image_gradient_weight
        if not math.isfinite(rank_weight) or rank_weight < 0 or not math.isfinite(rank_temperature) or rank_temperature <= 0:
            raise ValueError('Invalid rank contrast weight or temperature')
        self.rank_weight, self.rank_temperature = rank_weight, rank_temperature
        # Stable inverse softplus. The floor only handles an all-zero training median.
        magnitude = max(initial_magnitude, 1e-6)
        with torch.no_grad():
            self.sia_readout[-1].bias.copy_(self.sia_mean.new_tensor([
                magnitude + math.log(-math.expm1(-magnitude)), initial_phase]))

    def predict_polar(self, condition_features, clinical):
        return decode_polar(self.sia_readout(self.sia_features(condition_features, clinical)))

    def predict_sia(self, condition_features, clinical):
        return self.predict_polar(condition_features, clinical)[0]

    def forward(self, pre, post, sia_vector=None, clinical=None, mask=None, t=None,
                noise=None, sia_magnitude=None):
        batch = pre.shape[0]
        if sia_vector is None or sia_vector.shape != (batch, 2) or not torch.isfinite(sia_vector).all():
            raise ValueError('sia_vector must be finite [B,2] in physical D')
        t = torch.rand(batch, device=pre.device, dtype=pre.dtype) if t is None else t.to(pre)
        if t.shape != (batch,) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError('Invalid time')
        cond = self.condition_encoder(pre)
        if self.rank_weight:
            hidden = self.sia_readout[1](self.sia_readout[0](self.sia_features(cond, clinical)))
            q, magnitude, phase = decode_polar(self.sia_readout[2](hidden))
        else:
            q, magnitude, phase = self.predict_polar(cond, clinical)
        target = sia_vector.to(pre)
        tau = t[:, None, None, None]
        velocity = self.unet((1-tau)*pre+tau*post, t, cond, clinical=clinical)
        normalized_image_loss = masked_mse(velocity, post-pre, mask)
        image_loss = self.image_loss_multiplier*normalized_image_loss
        extra_logs = {}
        if self.image_gradient_weight:
            from model.spatial_gradient import masked_gradient_mse
            gradient_loss = masked_gradient_mse(velocity, post-pre, mask)
            image_loss = image_loss+self.image_gradient_weight*gradient_loss
            extra_logs['loss_image_gradient'] = gradient_loss
        magnitude_loss = F.l1_loss(magnitude/self.sia_ref, target.norm(dim=-1)/self.sia_ref)
        axis_loss = periodic_axis_loss(phase, target, self.axis_min_magnitude)
        output = dict(loss=image_loss+self.lambda_sia*magnitude_loss+self.lambda_axis*axis_loss,
                    loss_image=image_loss, loss_sia=magnitude_loss, loss_axis=axis_loss,
                    loss_image_normalized=normalized_image_loss,
                    velocity_pred=velocity, q_clean=q, sia_clean=magnitude, t=t, **extra_logs)
        if self.rank_weight:
            from model.rank_contrast import axial_rank_contrast
            auxiliary = axial_rank_contrast(hidden, target/self.sia_ref, self.rank_temperature)
            output.update(loss=output['loss']+self.rank_weight*auxiliary, loss_rank_contrast=auxiliary)
        return output
