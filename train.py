import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from vrp_env import VRPEnv
from model.VRPActor  import VRPActor
from model.VRPCritic import VRPCritic
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(actor, critic, n=20, capacity=30, batch_size=128, epochs=20, steps_per_epoch=500):
    """
    Boucle d'entraînement acteur critique par REINFORCE 
    Comme dans le papier, on prend le critique, on simule et on compare les deux
    Critique = baseline
    B : taille de batches
    N : nombre de points dans le pb
    D : dimension de l'espace des RNN et des embeddings

    BEAUCOUP PLUS LOURD QUE CE QUE JE PENSAIS : 
    trop long ? 
    Petit topo des variables : 
        - static : (B, N+1, 2) coordonnées des points de livraison (+ 0,0 par exemple = dépôt)
        - dynamic : (B, N+1, 2) demandes restantes, charge restante du livreur
        - load : (B,) charge restante du livreur
        - cur : (B,) index du point où se trouve le livreur
        - mask : (B, N+1) masque des actions valides
        - h, c : (1, B, D) mémoire du RNN
        - active : (B, 1) les instances non terminées
        - done : (B,1) les instances qui viennent de terminer à une étape donnée
        - log_probs : (B, T) log proba des actions prises à chaque étape
        - total_dist : (B, 1) distance totale parcourue à la fin de l'épisode
        - R : (B, 1) reward totale
        - baseline : (B, 1) reward estimé par le critique
        - adv : (B, 1) avantage (ou désavantage) pour l'acteur
        - Autres ... ? Faut que je refasse le tour
    A chaque étape :
        - on récupère le masque des actions valides
        - on fait un pas avec l'acteur pour récupérer les proba d'action
        - on échantillonne une action (pas de greedy pendant l'entraînement)
        -> un peu flou, on nous propose de choisir selon une distribution
        -> greedy pourrait s'appliquer, mais pour max l'exploration je prends pas
    """
    actor.to(DEVICE).train()
    critic.to(DEVICE).train()
    opt_a = optim.Adam(actor.parameters(),  lr=1e-4)
    opt_c = optim.Adam(critic.parameters(), lr=1e-4)
    env   = VRPEnv(n, capacity, batch_size, DEVICE)

    for epoch in range(epochs):
        actor.train()
        epoch_dist, epoch_la, epoch_lc = [], [], []
        epoch_ent = []
        for _ in range(steps_per_epoch):     
            entropies = []  
            static, dynamic = env.reset()
            B = batch_size

            h, c = actor.init_hidden(B, actor.D)
            h, c = h.to(DEVICE), c.to(DEVICE)
            cur = torch.zeros(B, dtype=torch.long, device=DEVICE)
            mask0 = env.get_mask()
            # Un pas dans le vide pour setup la baseline
            # Peut-être pas le plus propre ? Suffisant.
            baseline = critic(static, dynamic)


            h, c = actor.init_hidden(B, actor.D)
            h, c = h.to(DEVICE), c.to(DEVICE)
            cur = torch.zeros(B, dtype=torch.long, device=DEVICE)
            log_probs = []
            total_dist = torch.zeros(B, device=DEVICE)
            active = torch.ones(B, dtype=torch.bool, device=DEVICE)

            MAX_STEPS = n * (int(capacity) + 1)

            for _ in range(MAX_STEPS):
                mask = env.get_mask()

                probs, (h, c) = actor.step(static, dynamic, cur, (h, c), mask)
                #Categorical permet de pouvoir sample selon une distrib
                dist_cat = Categorical(probs)
                action = dist_cat.sample()
                action = torch.where(active, action, torch.zeros_like(action))
            
                log_probs.append(dist_cat.log_prob(action) * active.float())
                entropy = Categorical(probs).entropy().mean().item()
                epoch_ent.append(entropy)
                entropies.append(dist_cat.entropy() * active.float()) 
                static, dynamic, r, done = env.step(action)
                total_dist += (-r) * active.float()

                active = active & ~done
                cur = action

                if not active.any():
                    break

            #ne devrait plus arriver, mais je garde les logs au cas où, c'est informatif
            unsatisfied = (env.demands[:, 1:] > 1e-4).any(dim=1)
            if unsatisfied.any():
                idx = unsatisfied.nonzero()[0].item()
                print("Demandes restantes :", env.demands[idx, 1:].tolist())
                print("Charge restante    :", env.load[idx].item())
                print("Masque             :", env.get_mask()[idx].tolist())
                print("Steps utilisés     :", _)
            R = -total_dist
            lp = torch.stack(log_probs, dim=1).sum(dim=1)
            # Losses du papier

            mean_entropy = torch.stack(entropies, dim=1).sum(dim=1).mean()
            adv = R - baseline.detach()
            loss_a = -(adv * lp).mean() - (0.01 * mean_entropy)
            loss_c = ((R.detach() - baseline) ** 2).mean()

            opt_a.zero_grad()
            loss_a.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 2)
            opt_a.step()

            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 2)
            opt_c.step()

            # On récup les stats
            # En faire un graphique à la fin ?
            # Actuellement ce qui est important c'est epoch dist
            # Mais à priori L actor ~ 0 et L critic vers 1 c'est pas mal
            epoch_dist.append(total_dist.mean().item())
            epoch_la.append(loss_a.item())
            epoch_lc.append(loss_c.item())

        print(f"Epoch {epoch+1:3d} | "
              f"dist={sum(epoch_dist)/len(epoch_dist):.4f} | "
              f"L_actor={sum(epoch_la)/len(epoch_la):.4f} | "
              f"L_critic={sum(epoch_lc)/len(epoch_lc):.4f} | "
              f"entropy={sum(epoch_ent)/len(epoch_ent):.4f}")
        actor.eval() # Coupe le dropout
        with torch.no_grad():
            eval_static, eval_dynamic = env.reset()
            h, c = actor.init_hidden(B, actor.D)
            h, c = h.to(DEVICE), c.to(DEVICE)
            cur = torch.zeros(B, dtype=torch.long, device=DEVICE)
            active = torch.ones(B, dtype=torch.bool, device=DEVICE)
            eval_dist = torch.zeros(B, device=DEVICE)

            for _ in range(MAX_STEPS):
                mask = env.get_mask()
                probs, (h, c) = actor.step(eval_static, eval_dynamic, cur, (h, c), mask)
                
                action = probs.argmax(dim=1) 
                action = torch.where(active, action, torch.zeros_like(action))
                
                eval_static, eval_dynamic, r, done = env.step(action)
                eval_dist += (-r) * active.float()
                active = active & ~done
                cur = action
                if not active.any():
                    break
                    
        print(f"Distance EVAL Greedy : {eval_dist.mean().item():.4f}")
        actor.train() # On remet en mode entraînement

if __name__ == '__main__':
    actor  = VRPActor(128)
    critic = VRPCritic(128)

    train(actor, critic, n=10, capacity=20, batch_size=128,
          epochs=30, steps_per_epoch=1000)
    torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict()}, "vrp_checkpoint.pt")
