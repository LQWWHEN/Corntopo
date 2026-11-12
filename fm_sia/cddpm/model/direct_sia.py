"""Image flow with a deterministic, preoperative-only SIA magnitude head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.fm_sia import JointFlowMatching
from model.fm_common import masked_mse, ode_step


class DirectSIAFlowMatching(JointFlowMatching):
    def __init__(self, condition_encoder, unet, sia_mean, sia_std, *, initial_sia,
                 sample_steps=100, method='euler', lambda_scalar=1.0, sia_ref=1.0):
        super().__init__(condition_encoder, unet, sia_mean, sia_std,
            sample_steps=sample_steps, method=method, sia_mode='scalar',
            lambda_scalar=lambda_scalar, lambda_sia=0.0, sia_ref=sia_ref)
        if unet.sia_interaction is not None:
            raise ValueError('Direct SIA uses an image-only UNet, without SIA flow interaction')
        if not math.isfinite(initial_sia) or initial_sia < 0:
            raise ValueError('Initial SIA must be a finite nonnegative training-fold median')
        clinical_width = (unet.clinical_encoder.feature_embedding.shape[1]
                          if unet.clinical_encoder is not None else 0)
        self.sia_readout = nn.Sequential(nn.Linear(512+clinical_width, 128),
                                        nn.SiLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.sia_readout[-1].weight)
        # Stable inverse softplus. A zero training median starts near zero.
        value = max(float(initial_sia), 1e-4)
        nn.init.constant_(self.sia_readout[-1].bias, value+math.log(-math.expm1(-value)))

    def predict_sia(self, condition_features, clinical):
        feature = condition_features[-1].mean(dim=(-2, -1))
        if self.unet.clinical_encoder is not None:
            embedding, _ = self.unet.clinical_encoder(clinical)
            feature = torch.cat((feature, embedding), dim=1)
        return F.softplus(self.sia_readout(feature).squeeze(-1))

    def forward(self, pre, post, sia_vector=None, clinical=None, mask=None, t=None,
                noise=None, sia_magnitude=None):
        batch = pre.shape[0]
        if (sia_magnitude is None or sia_magnitude.shape not in ((batch,), (batch, 1))
                or not torch.isfinite(sia_magnitude).all() or (sia_magnitude < 0).any()):
            raise ValueError('sia_magnitude must be finite nonnegative [B] or [B,1] in D')
        t = torch.rand(batch, device=pre.device, dtype=pre.dtype) if t is None else t.to(pre)
        if t.shape != (batch,) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
            raise ValueError('Invalid time')
        cond = self.condition_encoder(pre)
        # This readout never receives post, interpolated image, time or noise.
        sia = self.predict_sia(cond, clinical)
        tau = t[:, None, None, None]
        velocity = self.unet((1-tau)*pre+tau*post, t, cond, clinical=clinical)
        image_loss = masked_mse(velocity, post-pre, mask)
        scalar_loss = F.l1_loss(sia/self.sia_ref, sia_magnitude.to(pre).reshape(-1)/self.sia_ref)
        return dict(loss=image_loss+self.lambda_scalar*scalar_loss,
                    loss_image=image_loss, loss_scalar=scalar_loss, loss_sia=scalar_loss,
                    velocity_pred=velocity, sia_clean=sia, t=t)

    @torch.no_grad()
    def sample(self, pre, clinical=None, noise=None, steps=None, method=None):
        steps = self.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError('steps must be positive')
        cond = self.condition_encoder(pre)
        sia = self.predict_sia(cond, clinical)
        def field(state, time):
            t = pre.new_full((pre.shape[0],), time)
            return (self.unet(state[0], t, cond, clinical=clinical),)
        state = (pre,)
        for k in range(steps):
            state = ode_step(field, state, k/steps, 1/steps, method or self.method)
        return dict(pred=state[0], sia=sia, sia_unclipped=sia,
                    sia_state=(sia[:, None]-self.sia_mean)/self.sia_std)
