
import torch
import torch.nn as nn
import math
from model.resblock import ResBlock,UpSample,DownSample
from model.attention import AttentionBlock


class ClinicalConditionEncoder(nn.Module):
    """Encode standardized scalar measurements as a vector and feature tokens."""

    def __init__(self, num_features, embed_dim=128):
        super().__init__()
        if num_features < 1:
            raise ValueError("Clinical conditioning requires at least one feature")
        self.num_features = num_features
        self.global_encoder = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.scalar_projection = nn.Linear(1, embed_dim)
        self.feature_embedding = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
        self.token_norm = nn.LayerNorm(embed_dim)

    def forward(self, clinical):
        if clinical is None or clinical.ndim != 2 or clinical.shape[1] != self.num_features:
            raise ValueError(f"clinical must have shape [batch, {self.num_features}]")
        global_embedding = self.global_encoder(clinical)
        feature_tokens = self.token_norm(
            self.scalar_projection(clinical.unsqueeze(-1))
            + self.feature_embedding.unsqueeze(0)
        )
        return global_embedding, feature_tokens


class ClinicalFusion(nn.Module):
    """Fuse tabular condition with a low-resolution image feature map."""

    METHODS = ("concat", "add", "self_attention", "cross_attention", "film")

    def __init__(self, channels, embed_dim, method, num_heads=4):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(f"Unknown clinical fusion method: {method}")
        self.method = method
        if method == "concat":
            self.projection = nn.Conv2d(channels + embed_dim, channels, 1)
        elif method == "add":
            self.projection = nn.Linear(embed_dim, channels)
        elif method == "film":
            self.affine = nn.Linear(embed_dim, channels * 2)
            nn.init.zeros_(self.affine.weight)
            nn.init.zeros_(self.affine.bias)
        else:
            if channels % num_heads:
                raise ValueError("Attention channels must be divisible by num_heads")
            self.image_norm = nn.LayerNorm(channels)
            self.clinical_norm = nn.LayerNorm(channels)
            self.clinical_projection = nn.Linear(embed_dim, channels)
            self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, image, global_embedding, feature_tokens):
        batch, channels, height, width = image.shape
        if self.method == "concat":
            broadcast = global_embedding[:, :, None, None].expand(-1, -1, height, width)
            return image + self.projection(torch.cat((image, broadcast), dim=1))
        if self.method == "add":
            return image + self.projection(global_embedding)[:, :, None, None]
        if self.method == "film":
            gamma, beta = self.affine(global_embedding).chunk(2, dim=1)
            return image * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        spatial = image.flatten(2).transpose(1, 2)
        query = self.image_norm(spatial)
        # The global token also gives cross attention two tokens with one scalar input.
        tokens = torch.cat((global_embedding.unsqueeze(1), feature_tokens), dim=1)
        tokens = self.clinical_norm(self.clinical_projection(tokens))
        if self.method == "self_attention":
            sequence = torch.cat((query, tokens), dim=1)
            update, _ = self.attention(sequence, sequence, sequence, need_weights=False)
            update = update[:, :height * width]
        else:
            update, _ = self.attention(query, tokens, tokens, need_weights=False)
        return image + update.transpose(1, 2).reshape(batch, channels, height, width)


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


    def __init__(self,time_dim=512,discrete=True,clinical_dim=0,
                 fusion_method="none",clinical_embed_dim=128,
                 sia_embed_dim=None, sia_num_heads=4, sia_state_dim=2):

        super().__init__()


        self.time_dim = time_dim
        self.time_emb = TimeEmbedding(discrete=discrete)
        self.sia_interaction = None
        if sia_embed_dim is not None:
            from model.sia_branch import SIAInteraction
            self.sia_interaction = SIAInteraction(512, time_dim, sia_embed_dim, sia_num_heads, sia_state_dim)
        if fusion_method != "none" and clinical_dim < 1:
            raise ValueError("Specify clinical_dim > 0 when clinical fusion is enabled")
        if fusion_method == "none":
            self.clinical_encoder = None
            self.clinical_fusion = None
        else:
            self.clinical_encoder = ClinicalConditionEncoder(clinical_dim, clinical_embed_dim)
            self.clinical_fusion = ClinicalFusion(512, clinical_embed_dim, fusion_method)
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



    def forward(self,x,t,cond,clinical=None,sia_state=None):

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
        if self.clinical_fusion is not None:
            if clinical is None or clinical.shape[0] != x4.shape[0]:
                raise ValueError("Clinical batch is missing or has a different batch size")
            embedding, tokens = self.clinical_encoder(clinical)
            x4 = self.clinical_fusion(x4, embedding, tokens)

        sia_velocity = None
        if self.sia_interaction is not None:
            if sia_state is None:
                raise ValueError("SIA UNet requires a standardized sia_state matching its state dimension")
            x4, sia_velocity = self.sia_interaction(x4, sia_state, t)

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

        image_velocity = self.out(x)
        if self.sia_interaction is not None:
            return image_velocity, sia_velocity
        return image_velocity

