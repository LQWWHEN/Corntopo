"""Time-dependent clinical FiLM on the preoperative image condition."""
from typing import NamedTuple

from torch import Tensor

from model.conditional_fno import ConditionalFNO


class ClinicalCondition(NamedTuple):
    image: Tensor
    gain: Tensor
    shift: Tensor

    def at_time(self, t):
        weight = 2 * (1 - t.reshape(-1, 1, 1, 1))
        return self.image * (1 + weight * self.gain) + weight * self.shift


class EarlyClinicalFNO(ConditionalFNO):
    def prepare_condition(self, condition_features, clinical=None):
        image = self.project_features(condition_features)
        embedding, _ = self.clinical_encoder(clinical)
        gain, shift = self.clinical_fusion.affine(embedding)[:, :, None, None].chunk(2, dim=1)
        return ClinicalCondition(image, gain, shift)

    def forward(self, x, t, cond, clinical=None):
        if not isinstance(cond, ClinicalCondition):
            cond = self.prepare_condition(cond, clinical)
        return self.conditioned_velocity(x, t, cond.at_time(t))
