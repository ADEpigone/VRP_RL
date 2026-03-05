import torch
import torch.nn as nn

from model.attention import GlimpseAttention

class VRPActor(nn.Module):
    """
    Modèle acteur, qui va prédire les trajectoires du livreur.
    B : taille de batches
    N : nombre de points dans le pb
    D : dimension de l'espace des RNN et des embeddings
    
    Petit topo des variables : 
     - static : (B, N+1, 2) coordonnées des points de livraison (+ 0,0 par exemple = dépôt)
     - dynamic : (B, N+1, 2) demandes restantes, charge restante du livreur
     - cur_node : (B, 1) noeud courant du livreur
     - hidden : (B,D)^2 état caché / mémoire du LSTM
     - h,c : (B,D)^2 nouvelle mémoire
     - p : (B,N+1) les probas retournées

    L'architecture (déduite du papier) :
    - embeds en conv1D
    - LSTM 1 couche en 128
    - Attention "glimpse", similaire mais pas égale à l'attention classique

    L'intuition derrière : 
    LSTM -> encode ce qu'on a fait 
    Attention -> Encode ce qu'on doit faire
    """
    def __init__(self, D):
        super().__init__()
        self.D = D

        self.static_emb = nn.Conv1d(2, D, 1)
        
        #Le papier suggérait qu'on traite les embeds en un bloc
        #Finalement le code offi part plutôt sur 2 embeds qu'on somme
        self.dem_emb = nn.Conv1d(1, D, 1)
        self.rem_emb = nn.Conv1d(1, D, 1)
        
        self.drop = nn.Dropout(0.1)
        self.lstm = nn.LSTMCell(D, D)
        self.att = GlimpseAttention(D)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_embeds(self, static, dynamic):
        s_bar = self.static_emb(static.permute(0, 2, 1)).permute(0, 2, 1)
        
        # [0:1] : la demande
        # [1:2] : la charge restante
        d_bar = self.dem_emb(dynamic[:, :, 0:1].permute(0, 2, 1)).permute(0, 2, 1)
        r_bar = self.rem_emb(dynamic[:, :, 1:2].permute(0, 2, 1)).permute(0, 2, 1)
        
        return s_bar, s_bar + d_bar + r_bar

    def init_hidden(self, B, D):
        h = torch.zeros(B, D)
        c = torch.zeros(B, D)
        return (h, c)
    
    def step(self, static, dynamic, cur_node, hidden, mask):
        s_bar, x_bar = self._get_embeds(static, dynamic)

        idx = cur_node.view(-1,1,1).expand(-1,1,self.D)
        lstm_input = self.drop(s_bar.gather(1, idx).squeeze(1))
        h, c = self.lstm(lstm_input, hidden)
        p = self.att(x_bar, h, mask)
    
        return p, (h, c)