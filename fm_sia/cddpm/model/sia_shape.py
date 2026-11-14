"""Condition postoperative image flow on a predicted SIA quadrupole field."""
import torch
from torch import nn
from torch.nn import functional as F
from model.polar_sia import PolarSIAFlowMatching, periodic_axis_loss
from model.fm_common import masked_mse, ode_step


def quadrupole_field(q, height, width, pre_mask=None):
    """A smooth direction code, not a calibrated corneal-power reconstruction.

    Raw map rows increase upwards when rendered with origin='lower'. The
    polynomial equals r² |q| cos(2 theta - phase(q)) inside the unit disk.
    """
    if q.ndim != 2 or q.shape[1] != 2:
        raise ValueError('The quadrupole requires physical double-angle vectors [B,2]')
    yy, xx = torch.meshgrid(
        2*(torch.arange(height, device=q.device, dtype=q.dtype)+.5)/height-1,
        2*(torch.arange(width, device=q.device, dtype=q.dtype)+.5)/width-1, indexing='ij')
    field = (q[:, 0, None, None]*(xx.square()-yy.square())+
             q[:, 1, None, None]*2*xx*yy)[:, None]
    field = field*(xx.square()+yy.square() <= 1).to(q)
    if pre_mask is not None:
        if pre_mask.shape != field.shape:
            raise ValueError('Preoperative validity must match the spatial condition')
        field = field*pre_mask.to(q)
    return field


class ShapeConditionedPolarSIAFlowMatching(PolarSIAFlowMatching):
    uses_preoperative_mask = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.unet, 'prepare_condition'):
            raise ValueError('SIA spatial conditioning requires conditional FNO')
        # No extra RNG draw: all parent parameters and initial outputs are preserved.
        self.sia_shape_gain = nn.Parameter(torch.zeros(1, self.unet.time_projection.out_features, 1, 1))

    def image_condition(self, condition_features, clinical, q, pre_mask=None):
        prepared = self.unet.prepare_condition(condition_features, clinical)
        field = quadrupole_field(q/self.sia_ref, *prepared.shape[-2:], pre_mask)
        return prepared+self.sia_shape_gain*field, field

    def forward(self, pre, post, sia_vector=None, clinical=None, mask=None, t=None,
                noise=None, sia_magnitude=None, pre_mask=None):
        batch = pre.shape[0]
        if sia_vector is None or sia_vector.shape != (batch, 2) or not torch.isfinite(sia_vector).all():
            raise ValueError('sia_vector must be finite [B,2] in physical D')
        t = torch.rand(batch, device=pre.device, dtype=pre.dtype) if t is None else t.to(pre)
        if t.shape != (batch,) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError('Invalid time')
        cond = self.condition_encoder(pre)
        q, magnitude, phase = self.predict_polar(cond, clinical)
        prepared, field = self.image_condition(cond, clinical, q, pre_mask)
        tau = t[:, None, None, None]
        velocity = self.unet((1-tau)*pre+tau*post, t, prepared, clinical=clinical)
        normalized_image_loss = masked_mse(velocity, post-pre, mask)
        image_loss = self.image_loss_multiplier*normalized_image_loss
        target = sia_vector.to(pre)
        magnitude_loss = F.l1_loss(magnitude/self.sia_ref, target.norm(dim=-1)/self.sia_ref)
        axis_loss = periodic_axis_loss(phase, target, self.axis_min_magnitude)
        return dict(loss=image_loss+self.lambda_sia*magnitude_loss+self.lambda_axis*axis_loss,
                    loss_image=image_loss, loss_image_normalized=normalized_image_loss,
                    loss_sia=magnitude_loss, loss_axis=axis_loss, velocity_pred=velocity,
                    q_clean=q, sia_clean=magnitude, sia_condition=field, t=t)

    @torch.no_grad()
    def sample(self, pre, clinical=None, noise=None, steps=None, method=None, pre_mask=None):
        steps = self.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError('steps must be positive')
        cond = self.condition_encoder(pre)
        q = self.predict_sia(cond, clinical)
        prepared, _ = self.image_condition(cond, clinical, q, pre_mask)
        def field(state, time):
            return (self.unet(state[0], pre.new_full((pre.shape[0],), time), prepared, clinical=clinical),)
        state = (pre,)
        for k in range(steps):
            state = ode_step(field, state, k/steps, 1/steps, method or self.method)
        return dict(pred=state[0], sia_vector=q, sia=q.norm(dim=-1), sia_state=(q-self.sia_mean)/self.sia_std)
