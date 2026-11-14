"""Couple polar SIA prediction to a supervised preoperative-endpoint velocity."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.polar_sia import PolarSIAFlowMatching, decode_polar, periodic_axis_loss
from model.fm_common import masked_mse, ode_step


def velocity_moments(value, pre_mask=None):
    """Twenty angular moments on preoperative-valid pixels in fixed radial bands."""
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError('Velocity descriptors require [B,1,H,W] FRONT maps')
    h, w = value.shape[-2:]
    yy, xx = torch.meshgrid(
        1-2*(torch.arange(h, device=value.device, dtype=value.dtype)+.5)/h,
        2*(torch.arange(w, device=value.device, dtype=value.dtype)+.5)/w-1, indexing='ij')
    radius = torch.sqrt(xx.square()+yy.square())
    angle = torch.atan2(yy, xx)
    noncentral = (radius > 0).to(value)
    angular = torch.stack((torch.ones_like(angle),
                          math.sqrt(2)*angle.cos()*noncentral,
                          math.sqrt(2)*angle.sin()*noncentral,
                          math.sqrt(2)*(2*angle).cos()*noncentral,
                          math.sqrt(2)*(2*angle).sin()*noncentral))
    rings = torch.stack([(radius >= i/4) &
                         (radius < (i+1)/4 if i < 3 else radius <= 1) for i in range(4)]).to(value)
    if pre_mask is not None and pre_mask.shape != value.shape:
        raise ValueError('Preoperative mask must match the velocity map')
    valid = torch.ones_like(value[:, 0]) if pre_mask is None else pre_mask[:, 0].to(value)
    support = valid[:, None]*rings[None]
    weights = support[:, :, None]*angular[None, None]
    weights = weights/support.sum((-2, -1)).clamp_min(1)[:, :, None, None, None]
    return torch.einsum('bchw,brkhw->brkc', value, weights).flatten(1)


class VelocityPolarSIAFlowMatching(PolarSIAFlowMatching):
    uses_preoperative_mask = True

    def __init__(self, *args, image_scale, endpoint_image_weight=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if (not math.isfinite(image_scale) or image_scale <= 0
                or not math.isfinite(endpoint_image_weight) or not 0 < endpoint_image_weight < 1):
            raise ValueError('Invalid physical feature scale or endpoint image weight')
        if not hasattr(self.unet, 'prepare_condition'):
            raise ValueError('Velocity-coupled SIA requires the conditional FNO')
        self.velocity_feature_scale = image_scale/self.sia_ref
        self.endpoint_image_weight = endpoint_image_weight
        # Extend the existing readout without changing parent weights or RNG state.
        layer = self.sia_readout[0]
        layer.weight = nn.Parameter(torch.cat((layer.weight.detach(), layer.weight.new_zeros(layer.out_features, 20)), dim=1))
        layer.in_features += 20

    def predict_polar(self, condition_features, clinical, initial_velocity=None, pre_mask=None):
        if initial_velocity is None:
            raise ValueError('The preoperative-only initial velocity is required')
        feature = torch.cat((self.sia_features(condition_features, clinical),
                             velocity_moments(initial_velocity*self.velocity_feature_scale, pre_mask)), dim=1)
        return decode_polar(self.sia_readout(feature))

    def predict_sia(self, condition_features, clinical, initial_velocity=None, pre_mask=None):
        return self.predict_polar(condition_features, clinical, initial_velocity, pre_mask)[0]

    def forward(self, pre, post, sia_vector=None, clinical=None, mask=None, t=None,
                noise=None, sia_magnitude=None, pre_mask=None):
        batch = pre.shape[0]
        if sia_vector is None or sia_vector.shape != (batch, 2) or not torch.isfinite(sia_vector).all():
            raise ValueError('sia_vector must be finite [B,2] in physical D')
        t = torch.rand(batch, device=pre.device, dtype=pre.dtype) if t is None else t.to(pre)
        if t.shape != (batch,) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError('Invalid time')
        cond = self.condition_encoder(pre)
        prepared = self.unet.prepare_condition(cond, clinical)
        initial_velocity = self.unet(pre, torch.zeros_like(t), prepared, clinical=clinical)
        q, magnitude, phase = self.predict_polar(cond, clinical, initial_velocity, pre_mask)
        tau = t[:, None, None, None]
        velocity = self.unet((1-tau)*pre+tau*post, t, prepared, clinical=clinical)
        random_loss = masked_mse(velocity, post-pre, mask)
        initial_loss = masked_mse(initial_velocity, post-pre, mask)
        alpha = self.endpoint_image_weight
        image_loss = (1-alpha)*random_loss+alpha*initial_loss
        target = sia_vector.to(pre)
        magnitude_loss = F.l1_loss(magnitude/self.sia_ref, target.norm(dim=-1)/self.sia_ref)
        axis_loss = periodic_axis_loss(phase, target, self.axis_min_magnitude)
        return dict(loss=image_loss+self.lambda_sia*magnitude_loss+self.lambda_axis*axis_loss,
                    loss_image=image_loss, loss_image_normalized=random_loss,
                    loss_image_initial=initial_loss, loss_sia=magnitude_loss, loss_axis=axis_loss,
                    velocity_pred=velocity, velocity_initial=initial_velocity,
                    q_clean=q, sia_clean=magnitude, t=t)

    @torch.no_grad()
    def sample(self, pre, clinical=None, noise=None, steps=None, method=None, pre_mask=None):
        steps = self.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError('steps must be positive')
        cond = self.condition_encoder(pre)
        prepared = self.unet.prepare_condition(cond, clinical)
        initial_velocity = self.unet(pre, pre.new_zeros(pre.shape[0]), prepared, clinical=clinical)
        q = self.predict_sia(cond, clinical, initial_velocity, pre_mask)
        def field(state, time):
            # Reuse the same first field evaluation for SIA and ODE integration.
            velocity = initial_velocity if time == 0 else self.unet(
                state[0], pre.new_full((pre.shape[0],), time), prepared, clinical=clinical)
            return (velocity,)
        state = (pre,)
        for k in range(steps):
            state = ode_step(field, state, k/steps, 1/steps, method or self.method)
        return dict(pred=state[0], sia_vector=q, sia=q.norm(dim=-1), sia_state=(q-self.sia_mean)/self.sia_std)
