import math
import torch
import torch.nn as nn

FOURIER_L = 3


def fourier_encode(coords):
    """coords (..., 2) -> (..., 2 + 4*FOURIER_L)"""
    freqs = (2.0 ** torch.arange(FOURIER_L, dtype=coords.dtype, device=coords.device)) * math.pi
    x_freq = coords[..., :1] * freqs
    y_freq = coords[..., 1:] * freqs
    return torch.cat([coords, x_freq.sin(), x_freq.cos(), y_freq.sin(), y_freq.cos()], dim=-1)


SPATIAL_FEAT = 2 + 4 * FOURIER_L   # 14
NODE_FEAT = SPATIAL_FEAT + 2        # 16


class NodeEncoder(nn.Module):
    """
    Encodeur classico
    """

    def __init__(self, d_model=128, n_heads=8, n_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(NODE_FEAT, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))

    def forward(self, x):
        return self.encoder(self.proj(x))
