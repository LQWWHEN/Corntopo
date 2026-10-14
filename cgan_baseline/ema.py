"""Same state-dict EMA rule as the current FM_SIA framework."""
import copy
import torch


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        self.ema_model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for key, value in self.ema_model.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source[key], alpha=1-self.decay)
            else:
                value.copy_(source[key])
