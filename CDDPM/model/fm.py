import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils import ModelWrapper


class FlowMatching(nn.Module):

    def __init__(self, condition_encoder, unet,method="euler"):
        super().__init__()

        self.condition_encoder = condition_encoder
        self.unet = unet

        # official FM path
        scheduler = CondOTScheduler()
        self.path = AffineProbPath(scheduler=scheduler)
        self.method=method

    def forward(self,pre,post,clinical=None):

        B = pre.shape[0]
        device = pre.device

        # continuous time
        t = torch.rand(B,device=device)
        path_sample = self.path.sample(t=t,x_0=pre,x_1=post)
        xt = path_sample.x_t
        velocity = path_sample.dx_t



        # condition
        cond = self.condition_encoder(pre)

        # predict velocity
        velocity_pred = self.unet(xt, t,cond,clinical=clinical)
        loss = F.mse_loss(velocity_pred,velocity)

        return {
            "loss":loss,
            "velocity_pred":velocity_pred,
            "velocity":velocity,
            "xt":xt,
            "t":t
        }



    @torch.no_grad()
    def sample(self,pre,steps=50,method=None,clinical=None):
        cond = self.condition_encoder(pre)
        device=pre.device


        # velocity wrapper

        def velocity_field(x,t):
            B=x.shape[0]
            if not torch.is_tensor(t):
                t=torch.tensor(t, device=x.device)
            if t.ndim==0:
                t=t.repeat(B)
            v=self.unet(x, t,cond,clinical=clinical)
            return v

        class WrappedModel(ModelWrapper):

            def forward(
                self,
                x,
                t,
                **extras
            ):

                return self.model(
                    x,
                    t,
                    **extras
                )



        wrapped_vf=WrappedModel(velocity_field)



        solver=ODESolver(velocity_model=wrapped_vf)


        if method is None:
            method = self.method
        # integrate:
        # x(0)=pre
        # x(1)=post
        time_grid=torch.linspace(
            0,
            1,
            steps+1,
            device=device
        )
        result = solver.sample(
            x_init=pre,
            time_grid=time_grid,
            step_size=None,
            method=method
        )

        return result

    def inference(self,x,clinical=None):
        return self.sample(x,clinical=clinical)
