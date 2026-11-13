"""Preoperative spatial-moment readout of the physical double-angle SIA vector."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.fm_sia import JointFlowMatching
from model.fm_common import masked_mse, ode_step


def axial_moments(feature):
    """Mean, x²-y² and 2xy moments, normalized on a centered circular aperture.

    x points right and y points up. The two quadratic weights are the real
    and imaginary parts of r² exp(2i theta). This readout preserves spatial
    orientation information; the ordinary CNN itself is not equivariant.
    """
    h, w = feature.shape[-2:]
    yy, xx = torch.meshgrid(1-2*(torch.arange(h, device=feature.device, dtype=feature.dtype)+.5)/h,
                            2*(torch.arange(w, device=feature.device, dtype=feature.dtype)+.5)/w-1, indexing='ij')
    aperture = (xx.square()+yy.square() <= 1).to(feature.dtype)
    weights = torch.stack((aperture, (xx.square()-yy.square())*aperture, 2*xx*yy*aperture))
    weights = weights / weights.abs().sum((-2, -1), keepdim=True).clamp_min(1)
    return torch.einsum('bchw,khw->bkc', feature, weights).flatten(1)


class AxialSIAFlowMatching(JointFlowMatching):
    def __init__(self, condition_encoder, unet, sia_mean, sia_std, *, feature_fusion='lab',
                 clinical_vector_indices=None, clinical_vector_mean=None, clinical_vector_std=None,
                 image_loss_units='normalized', image_scale=1., **kwargs):
        super().__init__(condition_encoder, unet, sia_mean, sia_std, sia_mode='vector', **kwargs)
        if image_loss_units not in ('normalized', 'diopter') or not math.isfinite(image_scale) or image_scale <= 0:
            raise ValueError('Invalid image loss units or training-fold image scale')
        self.image_loss_multiplier = (image_scale/self.sia_ref)**2 if image_loss_units == 'diopter' else 1.
        if feature_fusion not in ('lab', 'preop_frame'):
            raise ValueError('Unknown axial feature fusion')
        self.feature_fusion = feature_fusion
        if feature_fusion == 'preop_frame':
            if clinical_vector_indices is None or len(clinical_vector_indices) != 2:
                raise ValueError('Preoperative frame needs two ordered clinical vector indices')
            self.clinical_vector_indices = list(clinical_vector_indices)
            self.register_buffer('pre_vector_mean', torch.as_tensor(clinical_vector_mean, dtype=torch.float32))
            self.register_buffer('pre_vector_std', torch.as_tensor(clinical_vector_std, dtype=torch.float32))
        if unet.sia_interaction is not None:
            raise ValueError('Axial direct SIA requires an image-only UNet')
        clinical_width = (unet.clinical_encoder.feature_embedding.shape[1]
                          if unet.clinical_encoder is not None else 0)
        self.sia_readout = nn.Sequential(nn.Linear(3*512+clinical_width, 128), nn.SiLU(), nn.Linear(128, 2))
        nn.init.zeros_(self.sia_readout[-1].weight)
        with torch.no_grad():
            self.sia_readout[-1].bias.copy_(self.sia_mean)

    def predict_sia(self, condition_features, clinical):
        if self.feature_fusion == 'preop_frame':
            pre_vector = clinical[:, self.clinical_vector_indices]*self.pre_vector_std+self.pre_vector_mean
            feature = preoperative_frame_moments(condition_features[-1], pre_vector)
        else:
            feature = axial_moments(condition_features[-1])
        if self.unet.clinical_encoder is not None:
            embedding, _ = self.unet.clinical_encoder(clinical)
            feature = torch.cat((feature, embedding), dim=1)
        return self.sia_readout(feature)

    def forward(self, pre, post, sia_vector=None, clinical=None, mask=None, t=None,
                noise=None, sia_magnitude=None):
        batch = pre.shape[0]
        if sia_vector is None or sia_vector.shape != (batch, 2) or not torch.isfinite(sia_vector).all():
            raise ValueError('sia_vector must be finite [B,2] in physical D')
        t = torch.rand(batch, device=pre.device, dtype=pre.dtype) if t is None else t.to(pre)
        if t.shape != (batch,) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError('Invalid time')
        cond = self.condition_encoder(pre)
        q = self.predict_sia(cond, clinical)
        target = sia_vector.to(pre)
        tau = t[:, None, None, None]
        velocity = self.unet((1-tau)*pre+tau*post, t, cond, clinical=clinical)
        normalized_image_loss = masked_mse(velocity, post-pre, mask)
        # A difference cancels the min-max offset. Convert velocity error back
        # to D before squaring; the reference D is shared with both SIA terms.
        image_loss = self.image_loss_multiplier*normalized_image_loss
        vector_loss = F.huber_loss(q/self.sia_ref, target/self.sia_ref, delta=self.huber_delta)
        magnitude_loss = F.l1_loss(q.norm(dim=-1)/self.sia_ref, target.norm(dim=-1)/self.sia_ref)
        return dict(loss=image_loss+self.lambda_vector*vector_loss+self.lambda_sia*magnitude_loss,
                    loss_image=image_loss, loss_vector=vector_loss, loss_sia=magnitude_loss,
                    loss_image_normalized=normalized_image_loss,
                    velocity_pred=velocity, q_clean=q, sia_clean=q.norm(dim=-1), t=t)

    @torch.no_grad()
    def sample(self, pre, clinical=None, noise=None, steps=None, method=None):
        steps = self.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError('steps must be positive')
        cond = self.condition_encoder(pre)
        q = self.predict_sia(cond, clinical)
        def field(state, time):
            t = pre.new_full((pre.shape[0],), time)
            return (self.unet(state[0], t, cond, clinical=clinical),)
        state = (pre,)
        for k in range(steps):
            state = ode_step(field, state, k/steps, 1/steps, method or self.method)
        return dict(pred=state[0], sia_vector=q, sia=q.norm(dim=-1), sia_state=(q-self.sia_mean)/self.sia_std)


def preoperative_frame_moments(feature, pre_vector):
    """Condition spatial quadrupole moments on the preoperative flat meridian.

    Raw map rows increase upwards (the renderer uses origin='lower'). Both
    angular moments use the SAME radial normalization, so a rotation is a
    rotation of a two-component quadrupole, not of differently scaled values.
    No image resampling or trainable parameters; world-frame tokens remain.
    """
    h, w = feature.shape[-2:]
    yy, xx = torch.meshgrid(2*(torch.arange(h, device=feature.device, dtype=feature.dtype)+.5)/h-1,
                            2*(torch.arange(w, device=feature.device, dtype=feature.dtype)+.5)/w-1, indexing='ij')
    r2 = xx.square()+yy.square()
    aperture = (r2 <= 1).to(feature.dtype)
    radial_norm = (r2*aperture).sum().clamp_min(1)
    weights = torch.stack((aperture/aperture.sum().clamp_min(1),
                           (xx.square()-yy.square())*aperture/radial_norm,
                           2*xx*yy*aperture/radial_norm))
    moments = torch.einsum('bchw,khw->bkc', feature, weights)
    mean, cosine, sine = moments.unbind(1)
    length = pre_vector.norm(dim=-1)
    known = length > 1e-6
    denom = length.clamp_min(1e-6)
    c = torch.where(known, pre_vector[:, 0]/denom, torch.ones_like(length))[:, None]
    s = torch.where(known, pre_vector[:, 1]/denom, torch.zeros_like(length))[:, None]
    return torch.stack((mean, c*cosine+s*sine, -s*cosine+c*sine), dim=1).flatten(1)
