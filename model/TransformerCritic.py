import torch
import torch.nn as nn

from model.transformer_utils import fourier_encode, NodeEncoder


class TransformerVRPCritic(nn.Module):
    """
    Critique avec encodage transformers
    Un peu plus lourd, beaucoup plus expressif à priori, mais pas forcément mieux que le classique (à voir)
    """
    def __init__(self, D=128, n_heads=8, n_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.encoder = NodeEncoder(D, n_heads, n_layers, d_ff, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(D, D), nn.GELU(),
            nn.Linear(D, D // 2), nn.GELU(),
            nn.Linear(D // 2, 1),
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):
        x = torch.cat([fourier_encode(static), dynamic], dim=-1)
        return self.mlp(self.encoder(x).mean(1)).squeeze(-1)
