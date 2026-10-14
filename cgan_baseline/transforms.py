"""Exact right-angle transforms. Labels use doubled-angle X/Y coordinates."""
import random
import torch


class DataTransform:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5, rotation=True):
        self.hflip_prob, self.vflip_prob, self.rotation = hflip_prob, vflip_prob, rotation

    def get_params(self, seed=None):
        rng = random if seed is None else random.Random(seed)
        return dict(hflip=rng.random() < self.hflip_prob,
                    vflip=rng.random() < self.vflip_prob,
                    angle=rng.choice([0, 90, 180, 270]) if self.rotation else 0)

    def __call__(self, x, params=None, seed=None):
        params = self.get_params(seed) if params is None else params
        if params['hflip']:
            x = torch.flip(x, (-1,))
        if params['vflip']:
            x = torch.flip(x, (-2,))
        return torch.rot90(x, params['angle']//90, (-2, -1)) if self.rotation else x

    @staticmethod
    def transform_vector(q, params):
        # Either image-axis reflection changes only Y. A 90-degree rotation
        # rotates the doubled-angle vector by 180 degrees (both signs change).
        q = q.clone()
        if bool(params['hflip']) != bool(params['vflip']):
            q[1] = -q[1]
        if (params['angle']//90) % 2:
            q = -q
        return q
