"""Point-token transformer: every grid point is one token.

A token is the point's 20 input frames of (u, v) plus a few per-window scalars, concatenated with a Fourier encoding of
the point's (x, y) in [-1, 1]^2. A linear + GELU embedding, a stack of pre-LayerNorm self-attention blocks over all
2048 points of the 32x64 grid, and an MLP head give 80 outputs per point: 20 frames x (u, v) of the mean and
20 frames x (u, v) of the log-variance."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """(..., d) -> (..., d + 2 * d * n_freq): the coordinates followed by sin and cos of 2^k * pi * x, k < n_freq."""

    def __init__(self, in_dim: int, n_freq: int):
        super().__init__()
        self.out_dim = in_dim * (1 + 2 * n_freq)
        self.register_buffer("bands", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)

    def forward(self, x):
        xb = x.unsqueeze(-2) * self.bands.view(1, -1, 1)                     # (..., n_freq, d)
        return torch.cat([x, torch.cat([xb.sin(), xb.cos()], -1).flatten(-2)], -1)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, xq, xk, xv):
        b, n, d = xq.shape
        h = self.heads
        q = self.q(xq).view(b, n, h, d // h).transpose(1, 2)
        k = self.k(xk).view(b, n, h, d // h).transpose(1, 2)
        v = self.v(xv).view(b, n, h, d // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out(o.transpose(1, 2).contiguous().view(b, n, d))


class Block(nn.Module):
    """Pre-LayerNorm transformer block. The MLP branch starts with its own LayerNorm, so two LayerNorms run back to back
    on that path."""

    def __init__(self, dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim))

    def forward(self, x):
        # norm1 is evaluated once per projection, as in the training runs behind the submission (same values; the
        # gradient reaches it through three separate paths)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PointTokenTransformer(nn.Module):
    def __init__(self, c_in: int, c_out: int, coord_dim: int = 2, d_model: int = 256, num_heads: int = 8,
                 num_layers: int = 8, ff_mult: int = 4, fourier_freqs: int = 1):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = FourierFeatures(coord_dim, fourier_freqs)
        # parameters are created in this order: embed, proj_in, proj_out, blocks, head (the seed -> initial weights
        # mapping depends on it)
        self.embed = nn.Sequential(nn.Linear(self.pos_enc.out_dim + c_in, d_model), nn.GELU())
        self.proj_in = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.proj_out = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.blocks = nn.ModuleList([Block(d_model, num_heads, ff_mult) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
                                  nn.Linear(d_model * ff_mult, c_out))

    def forward(self, pos, feat):
        """pos (B, N, coord_dim), feat (B, N, c_in) -> (B, N, c_out)"""
        x = self.proj_in(self.embed(torch.cat([self.pos_enc(pos), feat], -1)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.proj_out(x))


def build(cfg: dict, c_in: int, c_out: int) -> PointTokenTransformer:
    """cfg: the "model" entry of configs/model.json"""
    return PointTokenTransformer(c_in, c_out, d_model=cfg["d_model"], num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
                                 ff_mult=cfg["ff_mult"], fourier_freqs=cfg["fourier_freqs"])
