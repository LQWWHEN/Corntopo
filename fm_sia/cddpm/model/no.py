from neuralop.models import FNO,UNO

import torch.nn as nn

import torch.nn.functional as F
from model.fm_common import masked_mae

class OperatorModule(nn.Module):
    """Keep neuralop constructor metadata out of nested PyTorch tensor state."""
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state.pop('_metadata', None)
        return state

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Also accept older saves containing neuralop's top-level metadata key.
        state = state_dict.copy()
        if hasattr(state_dict, '_metadata'):
            state._metadata = state_dict._metadata
        state.pop('_metadata', None)
        return super().load_state_dict(state, strict=strict, assign=assign)


class FourierNeuralOperator(OperatorModule):

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
        post=None,
        mask=None
    ):
        pred=self.fno(pre)

        output={
            "pred":pred
        }
        if post is not None:
            loss=masked_mae(
                pred,
                post,
                mask
            )
            output["loss"]=loss

        return output
    
    def inference(self,x): 
        return self.forward(x)
    
class UnetNeuralOperator(OperatorModule):

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
        post=None,
        mask=None
    ):
        pred=self.uno(pre)

        output={
            "pred":pred
        }
        if post is not None:
            loss=masked_mae(
                pred,
                post,
                mask
            )
            output["loss"]=loss

        return output
    
    def inference(self,x): 
        return self.forward(x)
    
