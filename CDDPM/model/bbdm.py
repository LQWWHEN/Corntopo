import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusion.scheduler import BrownianScheduler


def extract(a, t, x_shape):
    """
    extract timestep coefficient
    """

    out = a.gather(0,t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape)-1)))



class BBDM(nn.Module):

    def __init__(
        self,
        condition_encoder,
        unet,
        timesteps=1000,
        objective="noise",
        max_var=1.0,
        eta=1.0,
        sample_steps = 50
    ):

        super().__init__()

        self.condition_encoder = condition_encoder

        self.unet = unet

        self.timesteps = timesteps

        self.objective = objective
        self.sample_steps = sample_steps
        self.eta = eta
        self.scheduler = BrownianScheduler(
            timesteps=timesteps,
            max_var=max_var
        )
        


    # =====================================================
    # Brownian Bridge forward process
    #
    # xt=(1-mt)x0+mt*y+sigma_t eps
    #
    # x0 : post
    # y  : pre
    # =====================================================

    def q_sample(self,x0,y,t,noise=None):

        if noise is None:
            noise=torch.randn_like(x0)

        m_t = extract(self.scheduler.m_t,t,x0.shape)


        sigma_t = extract(self.scheduler.sigma,t,x0.shape)


        xt = ((1-m_t)*x0+m_t*y+sigma_t*noise)

        if self.objective=="noise":
            target=noise
        elif self.objective=="grad":
            target=m_t*(y-x0)+sigma_t*noise
        elif self.objective=="ysubx":
            target=y-x0
        else:
            raise ValueError


        return xt,target



    # =====================================================
    # recover x0
    # =====================================================

    def predict_x0(self,xt,y,t,pred):

        if self.objective=="noise":
            m_t=extract(self.scheduler.m_t,t,xt.shape)
            sigma_t=extract(self.scheduler.sigma,t,xt.shape)
            x0=(xt-m_t*y-sigma_t*pred)/(1-m_t)
        elif self.objective=="grad":
            x0=xt-pred
        elif self.objective=="ysubx":
            x0=y-pred


        return x0



    # =====================================================
    # training
    # =====================================================

    def forward(self,pre_op, post_op):


        B=post_op.shape[0]

        device=post_op.device

        t=torch.randint(0,self.timesteps,(B,),device=device).long()

        xt,target=self.q_sample(
            post_op,
            pre_op,
            t
        )

        # condition encoder
        cond_feature=self.condition_encoder(pre_op)
        pred=self.unet(xt,t,cond_feature)
        loss=F.mse_loss(
            pred,
            target
        )

        return {
            "loss":loss,
            "objective_pred":pred,
            "target":target,
            "t":t
        }



    # =====================================================
    # posterior sampling
    # =====================================================

    @torch.no_grad()
    def p_sample(self,xt,y,cond,t,t_next):


        B=xt.shape[0]
        tt=torch.full(
            (B,),
            t,
            device=xt.device,
            dtype=torch.long
        )
        pred=self.unet(
            xt,
            tt,
            cond
        )
        x0=self.predict_x0(
            xt,
            y,
            tt,
            pred
        )
        if t_next < 0:
            return x0



        tn=torch.full(
            (B,),
            t_next,
            device=xt.device,
            dtype=torch.long
        )


        m_t=extract(
            self.scheduler.m_t,
            tt,
            xt.shape
        )


        m_nt=extract(
            self.scheduler.m_t,
            tn,
            xt.shape
        )


        var_t=extract(
            self.scheduler.variance,
            tt,
            xt.shape
        )


        var_nt=extract(
            self.scheduler.variance,
            tn,
            xt.shape
        )

        # posterior variance
        sigma2=(
            var_t
            -
            var_nt
            *
            ((1-m_t)/(1-m_nt))**2
        )*var_nt/var_t



        sigma=torch.sqrt(
            torch.clamp(
                sigma2,
                min=1e-8
            )
        )*self.eta


        noise=torch.randn_like(xt)


        mean=(

            (1-m_nt)*x0
            +
            m_nt*y
            +
            torch.sqrt(
                torch.clamp(
                    (var_nt-sigma2)/var_t,
                    min=1e-8
                )
            )
            *
            (xt-(1-m_t)*x0-m_t*y)
        )

        return mean+sigma*noise

    # =====================================================
    # sampling
    # =====================================================
    @torch.no_grad()
    def sample(
        self,
        pre_op,
    ):
        device=pre_op.device
        B=pre_op.shape[0]

        cond=self.condition_encoder(pre_op)
        # start from source image
        x=pre_op.clone()

       


        times=np.linspace(
            self.timesteps-1,
            0,
            self.sample_steps
        ).astype(np.int64).tolist()

        for i,t in enumerate(times):
            if i==len(times)-1:
                t_next=-1
            else:
                t_next=times[i+1]

            x=self.p_sample(x,pre_op,cond,t,t_next)



        return x
    
    def inference(self,x): 
        return self.sample(x)