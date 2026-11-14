"""Conditional Fourier velocity plus a mean-free local corneal correction."""
import torch
from torch import nn
from torch.nn import functional as F
from model.conditional_fno import ConditionalFNO
from model.polar_sia import PolarSIAFlowMatching
from model.fm_common import ode_step


def mean_free_correction(raw, pre_mask=None):
    """Project onto zero mean over the preoperative valid pixels only."""
    mask = torch.ones_like(raw) if pre_mask is None else pre_mask.to(raw)
    if mask.shape != raw.shape:
        raise ValueError('Preoperative mask must match the local velocity field')
    mean = (raw*mask).sum((-2, -1), keepdim=True)/mask.sum((-2, -1), keepdim=True).clamp_min(1)
    return (raw-mean)*mask


class LocalResidualFNO(ConditionalFNO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Maintained entry points construct on CPU. Keep all parent and later
        # SIA initializations unchanged while initializing the extra branch.
        with torch.random.fork_rng(devices=[]):
            self.local_fine = nn.Sequential(nn.Conv2d(65,16,3,padding=1),nn.SiLU(),
                                            nn.Conv2d(16,16,3,padding=1),nn.SiLU())
            self.local_coarse = nn.Sequential(nn.AvgPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.SiLU(),
                                              nn.Conv2d(32,32,3,padding=1),nn.SiLU())
            self.local_output = nn.Sequential(nn.Conv2d(48,16,3,padding=1),nn.SiLU(),
                                              nn.Conv2d(16,1,1,bias=False))
            nn.init.zeros_(self.local_output[-1].weight)

    def velocity_components(self, x, t, cond, clinical=None, pre_mask=None):
        feature = cond if torch.is_tensor(cond) else self.prepare_condition(cond,clinical)
        time = self.time_projection(self.time_emb(t))[:,:,None,None]
        inputs = torch.cat((x,feature+time),dim=1)
        global_velocity = self.fno(inputs)
        fine = self.local_fine(inputs)
        coarse = F.interpolate(self.local_coarse(fine),size=fine.shape[-2:],mode='bilinear',align_corners=False)
        raw = self.local_output(torch.cat((fine,coarse),dim=1))
        return global_velocity, mean_free_correction(raw,pre_mask)

    def forward(self, x, t, cond, clinical=None, pre_mask=None):
        global_velocity, correction = self.velocity_components(x,t,cond,clinical,pre_mask)
        return global_velocity+correction


class LocalResidualPolarSIAFlowMatching(PolarSIAFlowMatching):
    uses_preoperative_mask = True

    def image_velocity(self, state, time, condition, clinical, pre_mask=None):
        return self.unet(state,time,condition,clinical=clinical,pre_mask=pre_mask)

    @torch.no_grad()
    def sample(self, pre, clinical=None, noise=None, steps=None, method=None, pre_mask=None):
        steps = self.sample_steps if steps is None else steps
        if steps < 1:
            raise ValueError('steps must be positive')
        condition = self.condition_encoder(pre)
        q = self.predict_sia(condition,clinical)
        prepared = self.unet.prepare_condition(condition,clinical)
        def field(state,time):
            t = pre.new_full((pre.shape[0],),time)
            return (self.image_velocity(state[0],t,prepared,clinical,pre_mask),)
        state = (pre,)
        for k in range(steps):
            state = ode_step(field,state,k/steps,1/steps,method or self.method)
        return dict(pred=state[0],sia_vector=q,sia=q.norm(dim=-1),sia_state=(q-self.sia_mean)/self.sia_std)
