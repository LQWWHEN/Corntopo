import torch
import torch.nn as nn
import torch.nn.functional as F

class DownSample(nn.Module):

    def __init__(self,ch):

        super().__init__()
        self.conv = nn.Conv2d(
            ch,
            ch,
            3,
            stride=2,
            padding=1
        )

    def forward(self,x):
        return self.conv(x)

class UpSample(nn.Module):

    def __init__(self,ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ch,
            ch,
            4,
            stride=2,
            padding=1
        )

    def forward(self,x):

        return self.conv(x)

class ResBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim=None
    ):
        super().__init__()

        self.time_dim = time_dim


        self.norm1 = nn.GroupNorm(
            8,
            in_channels
        )

        self.act1 = nn.SiLU()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )


        if time_dim is not None:

            self.time_proj = nn.Linear(
                time_dim,
                out_channels
            )


        self.norm2 = nn.GroupNorm(
            8,
            out_channels
        )

        self.act2 = nn.SiLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1
        )


        if in_channels != out_channels:

            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                1
            )

        else:
            self.skip = nn.Identity()


    def forward(self,x,t=None):

        h = self.conv1(
            self.act1(
                self.norm1(x)
            )
        )


        if t is not None:
            emb = self.time_proj(t)
            emb = emb[:,:,None,None]
            h = h + emb


        h = self.conv2(
            self.act2(
                self.norm2(h)
            )
        )


        return h + self.skip(x)