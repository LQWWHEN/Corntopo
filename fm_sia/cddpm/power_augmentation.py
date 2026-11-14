"""Paired isotropic power offsets with matched preoperative Kmax conditioning."""
import math
import torch

PRE_KMAX = 'pre_GENERAL DATA_K Max (Front)'


class PairedSphericalShift:
    def __init__(self, maximum_D, image_scale, clinical_index, clinical_std):
        if (not all(math.isfinite(v) for v in (maximum_D,image_scale,clinical_std))
                or maximum_D < 0 or min(image_scale,clinical_std) <= 0 or clinical_index < 0):
            raise ValueError('Invalid spherical-shift units or clinical index')
        self.maximum_D = maximum_D
        self.image_scale = image_scale
        self.clinical_index = clinical_index
        self.clinical_std = clinical_std

    def __call__(self, sample):
        if self.maximum_D == 0:
            return sample
        # DataLoader workers are seeded by the maintained training pipeline.
        delta = torch.empty((),dtype=sample['input'].dtype).uniform_(-self.maximum_D,self.maximum_D)
        output = dict(sample)
        for key,mask in [('input','mask_pre'),('target','mask_post')]:
            output[key] = sample[key] + delta/self.image_scale*sample[mask]
        output['condition'] = sample['condition'].clone()
        output['condition'][self.clinical_index] += delta/self.clinical_std
        # A common spherical component changes neither astigmatic vector nor axis.
        return output
