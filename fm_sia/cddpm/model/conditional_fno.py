"""Multiscale preoperative conditioning for a Fourier FM velocity field."""
import torch
from torch import nn
from torch.nn import functional as F
from neuralop.models import FNO
from model.no import OperatorModule
from model.unet import ClinicalConditionEncoder, ClinicalFusion, TimeEmbedding


class ConditionalFNO(OperatorModule):
    def __init__(self, clinical_dim=0, clinical_embed_dim=128, fusion_method='none',
                 modes=(32,32), hidden_channels=64, n_layers=4, domain_padding=.05):
        super().__init__()
        self.sia_interaction = None
        self.time_emb = TimeEmbedding(discrete=False)
        self.clinical_encoder = (ClinicalConditionEncoder(clinical_dim,clinical_embed_dim)
                                 if fusion_method != 'none' else None)
        self.clinical_fusion = (ClinicalFusion(64,clinical_embed_dim,fusion_method)
                                if fusion_method != 'none' else None)
        self.time_projection = nn.Linear(512,64)
        self.condition_projection = nn.ModuleList(nn.Conv2d(c,16,1) for c in (128,256,512,512))
        self.fno = FNO(n_modes=tuple(modes),in_channels=65,out_channels=1,
                       hidden_channels=hidden_channels,n_layers=n_layers,
                       domain_padding=domain_padding,domain_padding_mode='symmetric')

    def prepare_condition(self, condition_features, clinical=None):
        """Reuse the same preoperative representation across all ODE time steps."""
        size = condition_features[0].shape[-2:]
        projected = [F.interpolate(layer(feature),size=size,mode='bilinear',align_corners=False)
                     for layer,feature in zip(self.condition_projection,condition_features)]
        feature = torch.cat(projected,dim=1)
        if self.clinical_encoder is not None:
            embedding,tokens = self.clinical_encoder(clinical)
            feature = self.clinical_fusion(feature,embedding,tokens)
        return feature

    def forward(self, x, t, cond, clinical=None):
        feature = cond if torch.is_tensor(cond) else self.prepare_condition(cond,clinical)
        time = self.time_projection(self.time_emb(t))[:,:,None,None]
        return self.fno(torch.cat((x,feature+time),dim=1))
