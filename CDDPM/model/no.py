from neuralop.models import FNO,UNO

import torch.nn as nn

import torch.nn.functional as F

class FourierNeuralOperator(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        modes=(16,16),
        hidden_channels=64,
        n_layers=4
    ):
        super().__init__()


        self.fno = FNO(
            n_modes=modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers
        )


    def forward(
        self,
        pre,
        post=None
    ):
        pred=self.fno(pre)

        output={
            "pred":pred
        }
        if post is not None:
            loss=F.l1_loss(
                pred,
                post
            )
            output["loss"]=loss

        return output
    
    def inference(self,x): 
        return self.forward(x)
    
class UnetNeuralOperator(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        modes=[[16,16],[16,16],[8,8],[16,16],[16,16]],
        hidden_channels=64,
        n_layers=5
    ):
        super().__init__()


        self.uno = UNO(
            uno_n_modes= modes,
            uno_out_channels=[32,64,128,64,32],
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            uno_scalings=[[1,1],[0.5,0.5],[0.5,0.5],[2,2],[2,2]],
            fno_skip="linear",
            channel_mlp_skip="linear"
        )


    def forward(
        self,
        pre,
        post=None
    ):
        pred=self.uno(pre)

        output={
            "pred":pred
        }
        if post is not None:
            loss=F.l1_loss(
                pred,
                post
            )
            output["loss"]=loss

        return output
    
    def inference(self,x): 
        return self.forward(x)
    