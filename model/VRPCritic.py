import torch
import torch.nn as nn

class VRPCritic(nn.Module):
    def __init__(self, D=128):
        """
        Critique déduit du papier :
        - On prend des static/dynamique (on embed à part de l'acteur)
        - On prend les éléments de proba de l'acteur
        -> prédiction de la reward finale comme baseline

        Topo des variables : 
         - static : (B, N+1, 2) coordonnées des points de livraison (+ 0,0 par exemple = dépôt)
         - dynamic : (B, N+1, 2) demandes restantes, charge restante du livreur
         - actor_probs : (B, N+1) proba d'aller à chaque point selon l'acteur

        Deux convs d'embed et une proj simple
        """
        super().__init__()
        self.D = D
        self.static_emb  = nn.Conv1d(2, D, 1)
        self.dynamic_emb = nn.Conv1d(2, D, 1)

        self.proj = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_embeds(self, static, dynamic):
        s_bar = self.static_emb(static.permute(0, 2, 1)).permute(0, 2, 1)
        d_bar = self.dynamic_emb(dynamic.permute(0, 2, 1)).permute(0, 2, 1)
        return s_bar + d_bar 

    def forward(self, static, dynamic, actor_probs):
        x_bar = self._get_embeds(static, dynamic)

        context = (actor_probs.unsqueeze(2) * x_bar).sum(dim=1)

        value = self.proj(context)
        return value.squeeze(-1)