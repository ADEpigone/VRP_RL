import torch


class VRPEnv:
    """
    Environnement pour le pb du VRP
    Le pb du VRP ici est celui de la tournée de véhicule :
     - Un livreur a un camion avec une certaine capacité
     - Il doit livrer des clients avec des demandes d'une MEME marchandise en quantité différentes
     - Il part du dépôt, finit au dépôt, doit y retourner quand plus de marchandise.
    Le but ici est de MINIMISER la distance parcourue par le livreur.

    Tout y est vectorisé pour permettre l'entraînement, batch 1 = inférence normale

    Etats :
      - static : (B, N+1, 2) coordonnées des points de livraison (+ 0,0 par exemple = dépôt)
      - dynamic (d) : (B, N+1) demandes restantes, charge restante du livreur
      - load : (B,1) charge restante du livreur
      - cur : (B,1) index du point où se trouve le livreur
    Variables :
      - n (ou N) : nombre de clients
      - B : taille de batch
      - base_static/demands pour pouvoir reset (mais que partiellement)
    """
    def __init__(self, n, capacity, batch_size=1, device="cpu"):
        self.n = n
        self.cap = capacity
        self.B = batch_size
        self.device = device
        self.base_static = None
        self.base_demands = None

    def reset(self, new_points=True, new_demands=True):
        """
        Génération/setup d'une nouvelle instance
        Avec new_points et new_demands pas forcément obligatoires :
         - On peut soit garder la topologie soit les demandes
         -> Permet par exemple de regarder l'adaptabilité du réseau en direct ! <- pas utile finalement.
        """
        if new_points or self.base_static is None:
            self.base_static = torch.rand(self.B, self.n + 1, 2, device=self.device)
        if new_demands or self.base_demands is None:
            raw = torch.randint(1, 10, (self.B, self.n), dtype=torch.float, device=self.device)
            self.base_demands = torch.cat([torch.zeros(self.B, 1, device=self.device), raw], dim=1)
        
        self.static = self.base_static.clone()
        self.demands = self.base_demands.clone()
        self.load = torch.full((self.B,), float(self.cap), device=self.device)
        self.cur = torch.zeros(self.B, dtype=torch.long, device=self.device)
        return self.static, self.dynamic()

    def dynamic(self):
        """
        Récupération du vecteur dynamique à partir des demandes
        """
        rem = torch.clamp(self.load.unsqueeze(1) - self.demands, min = 0)
        return torch.stack([self.demands, rem], dim=2)

    def get_mask(self):
        mask = self.demands == 0
        # Interdit les clients trop gros
        mask[:, 1:] |= self.demands[:, 1:] > self.load.unsqueeze(1)
        # Interdit les clients si vide
        mask[:, 1:] |= (self.load == 0).unsqueeze(1)

        # On interdit le dépôt UNIQUEMENT si on y est déjà et qu'il reste des clients
        at_depot = (self.cur == 0)
        unmet_demand = (self.demands[:, 1:].sum(dim=1) > 0)
        mask[:, 0] = at_depot & unmet_demand

        # S'il a fini (plus de demande), on force à rester au dépôt
        done = ~unmet_demand
        mask[done, 0] = False   
        mask[done, 1:] = True

        return mask

    def step(self, action):
        """
        Applique une action et retourne les nouveaux états, la récompense et si il faut arrêter
        
        Transitions comme dans le papier :
        d_i(t+1) = max(0, d_i(t) - load(t))
        load(t+1) = capacity si dépôt
                    load - d_i(t) sinon
        reward = - distance parcourue

        Le code ici est commenté car le fait que ce soit vectoriel peut déstabiliser
        """

        # On récupère quelques variables de base
        # - où l'on est, si on est au dépôt
        batch_idx = torch.arange(self.B, device=self.device)
        at_depot = action == 0

        # On calcule ce qu'on livre 
        # - clamp de sécurité, en pratique dans le masque on laisse pas passer
        delivered = torch.clamp(self.demands[batch_idx, action], max=self.load)
        delivered = delivered * (~at_depot).float()

        # On met à jour les demandes et la charge
        # Encore une fois en pratique on pourrait mettre la demandé à 0
        # Mais car clamp, on soustrait
        self.demands[batch_idx, action] -= delivered
        self.demands.clamp_(min=0)

        self.load = torch.where(
            at_depot,
            torch.full_like(self.load, float(self.cap)),
            self.load - delivered,
        )

        # On calcule la distance parcourue par le livreur
        prev_pos = self.static[batch_idx, self.cur]
        next_pos = self.static[batch_idx, action]
        dist = torch.norm(next_pos - prev_pos, dim=1)

        self.cur = action
        # On met done si on a finit ET qu'on est au dépôt
        # On force le livreur à revenir au dépôt
        # ça évite à avoir à le faire soit même, et ça rend plus fidèle à ce dont on s'attendrait
        #tentative de bugfix, == 0 initialement, mais une petite valeur pour fpe ?
        done = (self.demands[:, 1:].sum(1) < 1e-4) & (self.cur == 0)
        return self.static, self.dynamic(), -dist, done