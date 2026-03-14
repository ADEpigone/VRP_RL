import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_utils import fourier_encode, NodeEncoder


class TransformerVRPActor(nn.Module):
    """
    Modèle acteur mais sous TRANSFORMERS 
    B : taille de batches 
    D : dimension de l'espace des RNN et des embeddings

    Petit topo des varibles :
        - Bon c'est la même pour que le classique, c'est fait pour être drop in

    Les grosses différences sont :
    - Une attention à la transformers, sans glimpse
    - Des embeds qui passent par un encodeur transformers avec des fourier features (ce qui est pas mal pour faire low dim -> high dim, peut demander du tuning)  
    """
    def __init__(self, D=128, n_heads=8, n_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.D = D
        self.encoder = NodeEncoder(D, n_heads, n_layers, d_ff, dropout)
        self.cap_proj = nn.Linear(1, D)
        self.ctx_proj = nn.Linear(3 * D, D)
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self._clip = 10.0
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_hidden(self, B, D):
        z = torch.zeros(B, D)
        return z, z

    def step(self, static, dynamic, cur_node, hidden, mask):
        B, device = static.shape[0], static.device
        node_emb = self.encoder(torch.cat([fourier_encode(static), dynamic], dim=-1))
        idx = torch.arange(B, device=device)

        #Dans le papier ils font la somme
        #Ici j'ai fait le choix de faire une proj pour de vrai
        #Plus lourd, mais plus expressif à priori
        ctx = self.ctx_proj(torch.cat([
            node_emb.mean(1),
            node_emb[idx, cur_node],
            self.cap_proj(dynamic[:, 0, 1:2]),
        ], dim=-1))

        #Pointer attention "classique"
        #On fait QK et pas QKV car ce qui nous intéresse c'est justement
        # ce "spectre" de scores

        Q = self.W_q(ctx).unsqueeze(1)
        K = self.W_k(node_emb)

        logits = (Q @ K.transpose(-2, -1)).squeeze(1) / math.sqrt(self.D)

        logits = self._clip * torch.tanh(logits)
        logits = logits.masked_fill(mask, float("-inf"))
        return F.softmax(logits, dim=-1), hidden
