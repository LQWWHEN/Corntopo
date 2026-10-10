"""Two-dimensional evolving state; high-dimensional features are internal only."""
import torch
from torch import nn


class SIAInteraction(nn.Module):
    def __init__(self, channels=512, time_dim=512, embed_dim=32, num_heads=4):
        super().__init__()
        if embed_dim < 1 or num_heads < 1 or embed_dim % num_heads:
            raise ValueError("sia_embed_dim must be positive and divisible by sia_num_heads")
        self.state_proj = nn.Sequential(nn.Linear(2, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.time_proj = nn.Linear(time_dim, embed_dim)
        self.image_proj = nn.Conv2d(channels, embed_dim, 1)
        self.position_proj = nn.Linear(2, embed_dim, bias=False)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.image_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2*embed_dim),
                                nn.SiLU(), nn.Linear(2*embed_dim, embed_dim))
        self.film = nn.Linear(embed_dim, 2*channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.velocity = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim),
                                      nn.SiLU(), nn.Linear(embed_dim, 2))

    def forward(self, image, state, time_embedding):
        if state.shape != (image.shape[0], 2):
            raise ValueError("SIA state must have shape [B,2]")
        h, w = image.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=image.device, dtype=image.dtype),
                                torch.linspace(-1, 1, w, device=image.device, dtype=image.dtype), indexing="ij")
        position = torch.stack((xx, yy), dim=-1).reshape(1, h*w, 2)
        spatial = self.image_proj(image).flatten(2).transpose(1, 2)
        spatial = self.image_norm(spatial + self.position_proj(position))
        token = (self.state_proj(state) + self.time_proj(time_embedding)).unsqueeze(1)
        update, _ = self.attn(self.query_norm(token), spatial, spatial, need_weights=False)
        token = token + update
        token = token + self.ff(token)
        feature = token[:, 0]
        gamma, beta = self.film(feature).chunk(2, dim=-1)
        image = image * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return image, self.velocity(feature)
