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


def periodic_axis_loss(phase, target, minimum_magnitude, arc_distance=False, reliability_scale=0.):
    magnitude = target.norm(dim=-1)
    valid = (magnitude >= minimum_magnitude) & (magnitude > 1e-8)
    target_phase = torch.atan2(target[:, 1], target[:, 0])
    delta = phase - target_phase
    if arc_distance:
        # Match the cosine loss range [0, 2] on the shortest doubled-phase arc.
        error = torch.atan2(delta.sin(), delta.cos()).abs() * (2 / math.pi)
    else:
        error = 1 - torch.cos(delta)
    if not valid.any():
        return phase.sum()*0
    if reliability_scale:
        stable = magnitude[valid].clamp(max=reliability_scale)
        weights = (stable / stable.max()).square()
        return (weights * error[valid]).sum() / weights.sum()
    return error[valid].mean()


def radial_quartiles(median, offsets):
    lower = median * offsets[:, 0].sigmoid()
    upper = median * (1 + F.softplus(offsets[:, 1]) / math.log(2))
    return torch.stack((lower, upper), dim=-1)


def quartile_loss(quartiles, target):
    error = target[:, None] - quartiles
    levels = quartiles.new_tensor([.25, .75])
    return torch.maximum(levels * error, (levels - 1) * error).sum(dim=-1).mean()


class PolarSIAFlowMatching(AxialSIAFlowMatching):
    def __init__(self, *args, initial_magnitude, initial_phase, lambda_axis=1.,
                 axis_min_magnitude=.1, rank_weight=0., rank_temperature=.2,
                 image_gradient_weight=0., axis_gradient_projection=False,
                 learned_axis_concentration=False, axis_arc_distance=False, quartile_weight=0.,
                 axis_reliability_scale=0., zero_polar_bias=False, **kwargs):
        super().__init__(*args, **kwargs)
        if (not all(math.isfinite(v) for v in (initial_magnitude, initial_phase, lambda_axis, axis_min_magnitude))
                or initial_magnitude < 0 or lambda_axis <= 0 or axis_min_magnitude < 0):
            raise ValueError('Invalid polar initialization or axis objective')
        self.lambda_axis = lambda_axis
        self.axis_arc_distance = axis_arc_distance
        self.axis_reliability_scale = axis_reliability_scale
        self.quartile_weight = quartile_weight
        self.quartile_head = None
        if quartile_weight:
            # Start at half and twice the parent median without advancing its CPU RNG.
            with torch.random.fork_rng(devices=[]):
                self.quartile_head = torch.nn.Linear(self.sia_readout[-1].in_features, 2)
            torch.nn.init.zeros_(self.quartile_head.weight)
            torch.nn.init.zeros_(self.quartile_head.bias)
        self.axis_gradient_projection = axis_gradient_projection
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
            bias = self.sia_readout[-1].bias
            if zero_polar_bias:
                bias.zero_()
            else:
                bias.copy_(bias.new_tensor([magnitude + math.log(-math.expm1(-magnitude)), initial_phase]))
        self.axis_concentration_head = None
        if learned_axis_concentration:
            from model.axis_concentration import AxisConcentration
            self.axis_concentration_head = AxisConcentration()

    def predict_polar(self, condition_features, clinical):
        return decode_polar(self.sia_readout(self.sia_features(condition_features, clinical)))

    def predict_sia(self, condition_features, clinical):
        return self.predict_polar(condition_features, clinical)[0]

    def image_velocity(self, state, time, condition, clinical, pre_mask=None):
        return self.unet(state, time, condition, clinical=clinical)

    def forward(self, pre, post, sia_vector=None, clinical=None, mask=None, t=None,
                noise=None, sia_magnitude=None, pre_mask=None):
        batch = pre.shape[0]
        if sia_vector is None or sia_vector.shape != (batch, 2) or not torch.isfinite(sia_vector).all():
            raise ValueError('sia_vector must be finite [B,2] in physical D')
        t = torch.rand(batch, device=pre.device, dtype=pre.dtype) if t is None else t.to(pre)
        if t.shape != (batch,) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError('Invalid time')
        cond = self.condition_encoder(pre)
        if self.rank_weight or self.axis_gradient_projection or self.axis_concentration_head is not None or self.quartile_head is not None:
            hidden = self.sia_readout[1](self.sia_readout[0](self.sia_features(cond, clinical)))
            q, magnitude, phase = decode_polar(self.sia_readout[2](hidden))
        else:
            q, magnitude, phase = self.predict_polar(cond, clinical, **self.sia_mask_argument(pre_mask))
        target = sia_vector.to(pre)
        tau = t[:, None, None, None]
        change = post - pre
        state = tau * change if self.integrates_power_change else (1-tau)*pre+tau*post
        velocity = self.image_velocity(state, t, cond, clinical, pre_mask)
        normalized_image_loss = masked_mse(velocity, change, mask)
        image_loss = self.image_loss_multiplier*normalized_image_loss
        extra_logs = {}
        if self.image_gradient_weight:
            from model.spatial_gradient import masked_gradient_mse
            gradient_loss = masked_gradient_mse(velocity, post-pre, mask)
            image_loss = image_loss+self.image_gradient_weight*gradient_loss
            extra_logs['loss_image_gradient'] = gradient_loss
        magnitude_loss = F.l1_loss(magnitude/self.sia_ref, target.norm(dim=-1)/self.sia_ref)
        if self.quartile_head is not None:
            quartiles = radial_quartiles(magnitude, self.quartile_head(hidden))
            auxiliary = quartile_loss(quartiles / self.sia_ref, target.norm(dim=-1) / self.sia_ref)
            extra_logs.update(loss_radial_median=magnitude_loss, loss_radial_quartiles=auxiliary)
            magnitude_loss = magnitude_loss + self.quartile_weight * auxiliary
        axis_loss = periodic_axis_loss(phase, target, self.axis_min_magnitude, self.axis_arc_distance,
                                       self.axis_reliability_scale)
        if self.axis_concentration_head is not None:
            from model.axis_concentration import concentration_axis_loss
            concentration = self.axis_concentration_head(hidden)
            axis_loss = concentration_axis_loss(phase, target, concentration, self.axis_min_magnitude)
            extra_logs['axis_kappa_mean'] = concentration.detach().mean()
            extra_logs['axis_kappa_min'] = concentration.detach().min()
            extra_logs['axis_kappa_max'] = concentration.detach().max()
        output = dict(loss=image_loss+self.lambda_sia*magnitude_loss+self.lambda_axis*axis_loss,
                    loss_image=image_loss, loss_sia=magnitude_loss, loss_axis=axis_loss,
                    loss_image_normalized=normalized_image_loss,
                    velocity_pred=velocity, q_clean=q, sia_clean=magnitude, t=t, **extra_logs)
        if self.axis_gradient_projection and self.training and torch.is_grad_enabled():
            from model.axis_gradient import axis_gradient_surrogate
            surrogate, diagnostics = axis_gradient_surrogate(
                hidden, self.lambda_sia*magnitude_loss, self.lambda_axis*axis_loss)
            output.update(loss=output['loss']+surrogate, **diagnostics)
        if self.rank_weight:
            from model.rank_contrast import axial_rank_contrast
            auxiliary = axial_rank_contrast(hidden, target/self.sia_ref, self.rank_temperature)
            output.update(loss=output['loss']+self.rank_weight*auxiliary, loss_rank_contrast=auxiliary)
        return output


class PowerChangePolarSIA(PolarSIAFlowMatching):
    """Integrate power change from zero and add it to the measured preoperative map."""

    integrates_power_change = True


class MaskedImageConditionPolarSIA(PolarSIAFlowMatching):
    """Pass the preoperative mask to the image core during training and sampling."""
    uses_preoperative_mask = True

    def image_velocity(self, state, time, condition, clinical, pre_mask=None):
        return self.unet(state, time, condition, clinical=clinical, pre_mask=pre_mask)

    def prepare_image_condition(self, condition, clinical, pre_mask):
        return self.unet.prepare_condition(condition, clinical, pre_mask)
