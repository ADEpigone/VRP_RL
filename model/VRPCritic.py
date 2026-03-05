import torch
import torch.nn as nn

# Pareil que pour l'acteur
# Ici se trouve un bloc d'attention simplifié
# Dans le code officiel c'est insinué qu'on utilise les probas des acteurs etc
# EN REALITE : dans le code offi ils font pas ça du tout
# Ils ont deux versions d'attention, des lstm etc etc
# Bref. Le topo est globalement le même mais on récupère que le contexte
class CriticProcessBlock(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.W_q = nn.Linear(D, D, bias=False)
        self.v = nn.Parameter(torch.Tensor(1, 1, D))
        nn.init.xavier_uniform_(self.v)

    def forward(self, h, x_bar):
        q = self.W_q(h).unsqueeze(1)
        u = torch.sum(self.v * torch.tanh(q + x_bar), dim=-1)
        prob = torch.softmax(u, dim=1)
        context = torch.bmm(prob.unsqueeze(1), x_bar).squeeze(1)
        return context


class VRPCritic(nn.Module):
    def __init__(self, D=128):
        """
        Critique déduit du papier :
        - On prend des static/dynamique (on embed à part de l'acteur)
        - On prend les éléments de proba de l'acteur
        -> prédiction de la reward finale comme baseline

        Critique réelle : 
        - On prend seulement les static/dynamique (pas de proba de l'acteur, on fait une attention à part)
        - Un bloc d'attention à 1 itération pour regarder le graphe

        Topo des variables : 
         - static : (B, N+1, 2) coordonnées des points de livraison (+ 0,0 par exemple = dépôt)
         - dynamic : (B, N+1, 2) demandes restantes, charge restante du livreur
         - actor_probs : (B, N+1) proba d'aller à chaque point selon l'acteur

        Deux convs d'embed et une proj simple
        """
        super().__init__()
        self.D = D
        self.static_emb  = nn.Conv1d(2, D, 1)
        
        self.dem_emb = nn.Conv1d(1, D, 1)

        self.process_block = CriticProcessBlock(D)

        self.proj = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, 1)
        )

        for p in self.parameters():
            if p.dim() > 1 and not isinstance(p, nn.Parameter):
                nn.init.xavier_uniform_(p)

    def _get_embeds(self, static, dynamic):
        s_bar = self.static_emb(static.permute(0, 2, 1)).permute(0, 2, 1)
        
        d_bar = self.dem_emb(dynamic[:, :, 0:1].permute(0, 2, 1)).permute(0, 2, 1)
        return s_bar + d_bar 

    def forward(self, static, dynamic):
        x_bar = self._get_embeds(static, dynamic)
        B = static.size(0)

        h = torch.zeros(B, self.D, device=static.device)
        context = self.process_block(h, x_bar)

        value = self.proj(context)
        return value.squeeze(-1)