import torch
import torch.nn as nn


class GlimpseAttention(nn.Module):
    """
    Glimpse attention implémentée comme dans le papier
    B : taille de batches
    N : nombre de points dans le pb
    D : dimension de l'espace des RNN et des embeddings
    """
    def __init__(self, D, C=10.0):
        super().__init__()
        self.C = C

        self.va = nn.Linear(D, 1)
        self.vc = nn.Linear(D, 1)


        self.wa = nn.Linear(2*D, D)
        self.wc = nn.Linear(2*D, D)

    def forward(self, x_bar, h, mask):
        """
        x_bar = (sbar + dbar) -> (B, N, D) les inputs embeddées
        NOTE: Dans le papier on a xbar = (sbar,dbar) mais il n'est JAMAIS explicitement expliqué
              ce qu'on fait, car après x_bar est considéré comme un vecteur et non des paires de vecteurs
              ici je pars du principe qu'on a fait la somme, ou qu'on a embeds les deux en même temps
              mais on a alors pas de couples.
        h = ht -> (B, D) la mémoire du RNN
        mask : (B, N) masque des actions interdites

        Le reste : c'est des formules
        Mais ce qu'il faut en tirer :
        - a = poids d'attention 
        - c = vecteur de contexte
        -> on propage et on regarde la distrib finale
        """

        u = self.va(torch.tanh(
            self.wa(torch.cat((x_bar, h.unsqueeze(1).expand_as(x_bar)), dim=-1)))
            ).squeeze(-1)
        #u = u / (128)**0.5
        u = self.C * torch.tanh(u)
        u = u.masked_fill(mask, float('-inf'))
        
        a = torch.softmax(u, dim=1)
        #print(a.shape, x_bar.shape)
        c = torch.bmm(a.unsqueeze(1), x_bar)

        u_bar = self.vc(torch.tanh( 
            self.wc(torch.cat((x_bar, c.expand_as(x_bar)), dim=-1)))
            ).squeeze(-1)
        
        #u_bar = u_bar / (128)**0.5
        u_bar = self.C * torch.tanh(u_bar)
        
        u_bar = u_bar.masked_fill(mask, float('-inf'))
        p = torch.softmax(u_bar, dim=1)
        return p