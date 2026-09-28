import torch
import torch.nn as nn


class AttentionBlock(nn.Module):

    def __init__(
        self,
        channels,
        num_heads=4
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads


        self.norm = nn.GroupNorm(
            32,
            channels
        )


        self.qkv = nn.Conv1d(
            channels,
            channels * 3,
            kernel_size=1
        )


        self.proj = nn.Conv1d(
            channels,
            channels,
            kernel_size=1
        )


    def forward(self,x):

        """
        x:
        B,C,H,W
        """

        B,C,H,W=x.shape


        residual=x


        # normalization

        x=self.norm(x)


        # flatten spatial

        x=x.reshape(
            B,
            C,
            H*W
        )


        # qkv

        qkv=self.qkv(x)


        q,k,v=torch.chunk(
            qkv,
            3,
            dim=1
        )


        head_dim=C//self.num_heads


        q=q.reshape(
            B,
            self.num_heads,
            head_dim,
            H*W
        )


        k=k.reshape(
            B,
            self.num_heads,
            head_dim,
            H*W
        )


        v=v.reshape(
            B,
            self.num_heads,
            head_dim,
            H*W
        )


        # attention

        attn=torch.einsum(
            "bhcn,bhcm->bhnm",
            q,
            k
        )


        attn=attn/(head_dim**0.5)


        attn=torch.softmax(
            attn,
            dim=-1
        )


        out=torch.einsum(
            "bhnm,bhcm->bhcn",
            attn,
            v
        )


        out=out.reshape(
            B,
            C,
            H*W
        )


        out=self.proj(out)


        out=out.reshape(
            B,
            C,
            H,
            W
        )


        return residual+out