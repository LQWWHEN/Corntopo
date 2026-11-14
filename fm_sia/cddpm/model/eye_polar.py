"""Polar SIA prediction in a reversible eye-conditioned anatomical frame."""
import torch
from model.polar_sia import PolarSIAFlowMatching
from model.eye_frame import reflect_left_image, reflect_left_vector


class EyeFramePolarSIAFlowMatching(PolarSIAFlowMatching):
    def __init__(self, *args, eye_index, eye_mean, eye_std, **kwargs):
        super().__init__(*args, **kwargs)
        self.eye_index = eye_index
        self.register_buffer('eye_mean', torch.tensor(float(eye_mean)))
        self.register_buffer('eye_std', torch.tensor(float(eye_std)))
        # The parent initializes magnitude/phase from canonical training targets.
        # Cartesian canonical means must not overwrite the polar readout bias.

    def eye_sign(self, clinical):
        value = clinical[:, self.eye_index]*self.eye_std+self.eye_mean
        return torch.where(value < 0, -torch.ones_like(value), torch.ones_like(value))

    def forward(self, pre, post, sia_vector=None, clinical=None, mask=None, **kwargs):
        sign = self.eye_sign(clinical)
        result = super().forward(
            reflect_left_image(pre, sign), reflect_left_image(post, sign),
            reflect_left_vector(sia_vector, sign), clinical=clinical,
            mask=reflect_left_image(mask, sign) if mask is not None else None, **kwargs)
        result['velocity_pred'] = reflect_left_image(result['velocity_pred'], sign)
        result['q_clean'] = reflect_left_vector(result['q_clean'], sign)
        return result

    @torch.no_grad()
    def sample(self, pre, clinical=None, **kwargs):
        sign = self.eye_sign(clinical)
        result = super().sample(reflect_left_image(pre, sign), clinical=clinical, **kwargs)
        result['pred'] = reflect_left_image(result['pred'], sign)
        result['sia_vector'] = reflect_left_vector(result['sia_vector'], sign)
        result['sia_state'] = (result['sia_vector']-self.sia_mean)/self.sia_std
        return result
