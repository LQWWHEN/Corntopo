
import torch
import torch.nn as nn
import math
from model.resblock import ResBlock,UpSample,DownSample
from model.attention import AttentionBlock
class TimeEmbedding(nn.Module):

    def __init__(self,discrete,dim=128):

        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim,512),
            nn.SiLU(),
            nn.Linear(512,512)
        )
        self.discrete = discrete


    def forward(self,t):
        half = self.dim//2
        emb = math.log(10000) / (half-1)
        emb = torch.exp( torch.arange(half,device=t.device)*-emb)

        if self.discrete:
            emb = t[:,None] * emb[None,:]
        else:
            emb = t[:,None] * emb[None,:]*2*math.pi

        emb = torch.cat([torch.sin(emb),torch.cos(emb)],dim=1)

        return self.mlp(emb)

class ConditionalUNet(nn.Module):


    def __init__(self,time_dim=512,discrete=True):

        super().__init__()


        self.time_dim = time_dim
        self.time_emb = TimeEmbedding(discrete=discrete)
        # input

        self.in_conv = nn.Conv2d(1, 64, 3,  padding=1)


        # Down

        self.down1 = ResBlock(64,128,time_dim)
        self.pool1 = DownSample(128)
        
        self.down2 = ResBlock(128, 256, time_dim)

        self.pool2 = DownSample(256)

        self.down3 = ResBlock(256,512,time_dim)
        self.attn3 = AttentionBlock(512,num_heads=4)

        self.pool3 = DownSample(512)


        # Middle

        self.mid = ResBlock(512,512,time_dim)
        self.attn_m = AttentionBlock(512,num_heads=4)

        # Up

        self.up3 = UpSample(512)
        self.up_block3 = ResBlock(512+512,256,time_dim)

        self.up2 = UpSample(256)
        self.up_block2 = ResBlock(256+256,128,time_dim)


        self.up1 = UpSample(128)
        self.up_block1 = ResBlock(128+128,64,time_dim)


        self.out = nn.Conv2d(64,1,3,padding=1)



    def forward(self,x,t,cond):

        f1,f2,f3,f4 = cond
        t = self.time_emb(t)


        # encoder

        x = self.in_conv(x)
        x1 = self.down1(x,t)
        x1 = x1 + f1
        x2 = self.pool1(x1)
        
        x2 = self.down2(x2,t)
        x2 = x2 + f2
        x3 = self.pool2(x2)


        x3 = self.down3(x3,t)
        x3 = self.attn3(x3)
        x3 = x3 + f3
        x4 = self.pool3(x3)

        # middle

        x4 = self.mid(x4,t)
        x4 = self.attn_m(x4)

        x4 = x4 + f4

        # decoder

        x = self.up3(x4)
        x = torch.cat([x,x3],dim=1)
        x = self.up_block3(x,t)



        x = self.up2(x)
        x = torch.cat([x,x2],dim=1)
        x = self.up_block2(x,t)

        x = self.up1(x)
        x = torch.cat([x,x1],dim=1)
        x = self.up_block1(x,t)

        return self.out(x)