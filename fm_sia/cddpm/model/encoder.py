
import torch
import torch.nn as nn
import math
from model.resblock import ResBlock,UpSample,DownSample

class ConditionEncoder(nn.Module):

    def __init__(self):

        super().__init__()


        # input
        self.in_conv = nn.Conv2d(
            1,
            128,
            3,
            padding=1
        )


        # stage1
        self.enc1 = ResBlock(
            128,
            128,
            None
        )


        self.down1 = DownSample(
            128
        )
        # stage2

        self.enc2 = ResBlock(
            128,
            256,
            None
        )


        self.down2 = DownSample(
            256
        )


        # stage3

        self.enc3 = ResBlock(
            256,
            512,
            None
        )


        self.down3 = DownSample(
            512
        )


        # middle

        self.enc4 = ResBlock(
            512,
            512,
            None
        )



    def forward(self,x):


        x=self.in_conv(x)

        f1=self.enc1(x)
        # [B,128,128,128]
        x=self.down1(f1)

        f2=self.enc2(x)
        # [B,256,64,64]


        x=self.down2(f2)


        f3=self.enc3(x)
        # [B,512,32,32]


        x=self.down3(f3)


        f4=self.enc4(x)
        # [B,512,16,16]


        return f1,f2,f3,f4