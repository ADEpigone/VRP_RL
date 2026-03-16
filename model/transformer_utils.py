import math
import torch
import torch.nn as nn

# Nombre de fréquences utilisées pour l'encodage de Fourier des coordonnées
FOURIER_L = 3


def fourier_encode(coords):
    """Encodage positionnel !"""
    freqs = (2.0 ** torch.arange(FOURIER_L, dtype=coords.dtype, device=coords.device)) * math.pi
    x_freq = coords[..., :1] * freqs
    y_freq = coords[..., 1:] * freqs
    # On concat x, y bruts + sin/cos pour chaque fréquence sur x et y
    return torch.cat([coords, x_freq.sin(), x_freq.cos(), y_freq.sin(), y_freq.cos()], dim=-1)


# Taille du vecteur de features spatial après encodage de Fourier : 2 coords + 4*L sin/cos
SPATIAL_FEAT = 2 + 4 * FOURIER_L # 14
# On ajoute 2 features dynamiques (demande + charge restante)
NODE_FEAT = SPATIAL_FEAT + 2  # 16


class NodeEncoder(nn.Module):
    """
    Encodeur classique
    Projette les features de chaque noeud puis passe dans un Transformer Encoder
    Renvoie les embeddings contextualisés (B, N+1, d_model)
    """

    def __init__(self, d_model=128, n_heads=8, n_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        # Projection linéaire des features brutes 
        self.proj = nn.Linear(NODE_FEAT, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))

    def forward(self, x):
        return self.encoder(self.proj(x))
