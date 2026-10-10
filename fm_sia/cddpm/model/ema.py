# ema.py

import copy
import torch


class EMA:

    def __init__(self,
                 model,
                 decay=0.9999):

        self.decay = decay

        self.ema_model = copy.deepcopy(model)

        self.ema_model.eval()

        for p in self.ema_model.parameters():

            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):

        msd = model.state_dict()
        esd = self.ema_model.state_dict()

        for k in esd.keys():

            if esd[k].dtype.is_floating_point:

                esd[k].mul_(self.decay)

                esd[k].add_(
                    msd[k],
                    alpha=1-self.decay
                )

            else:

                esd[k].copy_(msd[k])