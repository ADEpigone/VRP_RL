import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions import Categorical
from tqdm import tqdm
from model.FourierActor import FourierVRPActor
from vrp_env import VRPEnv
from model.VRPActor  import VRPActor
from model.VRPCritic import VRPCritic
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def crossing_penalty(coords, actions_list):
    """
    Compte les croisements d'arêtes intra-tour (entre deux passages au dépôt)
    coords: (B, N+1, 2)
    actions_list: liste de tenseurs (B,)
    """
    B = coords.shape[0]
    device = coords.device
    depot = torch.zeros(B, 1, dtype=torch.long, device=device)
    actions_t = torch.stack(actions_list, dim=1)
    path_idx = torch.cat([depot, actions_t], dim=1)  # (B, T+1)
    T = path_idx.shape[1]
    if T < 4:
        return torch.zeros(B, device=device)

    path = torch.gather(coords, 1, path_idx.unsqueeze(-1).expand(-1, -1, 2))
    p1, p2 = path[:, :-1], path[:, 1:]  # (B, E, 2)
    E = p1.shape[1]
    if E < 2:
        return torch.zeros(B, device=device)

    # IDs de tour : on incrémente à chaque passage au dépôt
    tour_id = (path_idx == 0).cumsum(dim=1)[:, :-1]  # (B, E)
    same_tour = tour_id.unsqueeze(2) == tour_id.unsqueeze(1)  # (B, E, E)

    # Test d'intersection par produits vectoriels
    a, b = p1.unsqueeze(2), p2.unsqueeze(2)  # (B, E, 1, 2)
    c, d = p1.unsqueeze(1), p2.unsqueeze(1)  # (B, 1, E, 2)
    cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    d1 = cross(d - c, a - c)
    d2 = cross(d - c, b - c)
    d3 = cross(b - a, c - a)
    d4 = cross(b - a, d - a)
    intersect = (d1 * d2 < 0) & (d3 * d4 < 0)

    # Triangle inférieur, non-adjacent, même tour (intra-tour uniquement)
    idx = torch.arange(E, device=device)
    pair_mask = idx.unsqueeze(1) > idx.unsqueeze(0) + 1  # (E, E)
    return (intersect & pair_mask.unsqueeze(0) & same_tour).float().sum(dim=(1, 2))


def train(actor, critic, n=20, capacity=30, batch_size=128, epochs=20, steps_per_epoch=500, eval_batch_size=1000,want_cross = True, lambda_cross=0.1, OUTPUT_DIR="checkpoints"):
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
    target_lr = 1e-5
    decay_epochs = 30
    scheduler_a = optim.lr_scheduler.CosineAnnealingLR(
        opt_a, T_max=decay_epochs, eta_min=target_lr
    )
    scheduler_c = optim.lr_scheduler.CosineAnnealingLR(
        opt_c, T_max=decay_epochs, eta_min=target_lr
    )
    env = VRPEnv(n, capacity, batch_size, DEVICE)
    eval_env = VRPEnv(n, capacity, eval_batch_size, DEVICE)
    eval_env.reset(new_points=True, new_demands=True)
    max_steps = n * (int(capacity) + 1)
    checkpoint_dir = os.path.join("", OUTPUT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        actor.train()
        epoch_dist, epoch_la, epoch_lc = [], [], []
        epoch_ent, epoch_cross = [], []
        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False):
            entropies = []  
            static, dynamic = env.reset()
            coords = static  # sauvegarde des coordonnées pour le calcul des croisements
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
            all_actions = []
            total_dist = torch.zeros(B, device=DEVICE)
            active = torch.ones(B, dtype=torch.bool, device=DEVICE)

            for _ in range(max_steps):
                mask = env.get_mask()

                probs, (h, c) = actor.step(static, dynamic, cur, (h, c), mask)
                #Categorical permet de pouvoir sample selon une distrib
                dist_cat = Categorical(probs)
                action = dist_cat.sample()
                action = torch.where(active, action, torch.zeros_like(action))
            
                log_probs.append(dist_cat.log_prob(action) * active.float())
                all_actions.append(action)
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
            if want_cross:
                crossings = crossing_penalty(coords, all_actions)  # (B,)
                R_shaped = R - lambda_cross * crossings
            else:
                R_shaped = R

            lp = torch.stack(log_probs, dim=1).sum(dim=1)
            # Losses du papier + pénalité de croisements intra-tour

            
            mean_entropy = torch.stack(entropies, dim=1).sum(dim=1).mean()
            adv = R_shaped - baseline.detach()
            loss_a = -(adv * lp).mean()
            loss_c = ((R_shaped.detach() - baseline) ** 2).mean()

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
            if want_cross:
                epoch_cross.append(crossings.mean().item())
            else:
                epoch_cross.append(0.0)

        print(f"Epoch {epoch+1:3d} | "
              f"dist={sum(epoch_dist)/len(epoch_dist):.4f} | "
              f"L_actor={sum(epoch_la)/len(epoch_la):.4f} | "
              f"L_critic={sum(epoch_lc)/len(epoch_lc):.4f} | "
              f"entropy={sum(epoch_ent)/len(epoch_ent):.4f} | "
              f"cross={sum(epoch_cross)/len(epoch_cross):.2f}")
        actor.eval() # Coupe le dropout
        with torch.no_grad():
            # Eval sur un benchmark fixe pour comparer les epochs de facon stable.
            eval_static, eval_dynamic = eval_env.reset(new_points=False, new_demands=False)
            B_eval = eval_batch_size
            h, c = actor.init_hidden(B_eval, actor.D)
            h, c = h.to(DEVICE), c.to(DEVICE)
            cur = torch.zeros(B_eval, dtype=torch.long, device=DEVICE)
            active = torch.ones(B_eval, dtype=torch.bool, device=DEVICE)
            eval_dist = torch.zeros(B_eval, device=DEVICE)

            for _ in range(max_steps):
                mask = eval_env.get_mask()
                probs, (h, c) = actor.step(eval_static, eval_dynamic, cur, (h, c), mask)
                
                action = probs.argmax(dim=1) 
                action = torch.where(active, action, torch.zeros_like(action))
                
                eval_static, eval_dynamic, r, done = eval_env.step(action)
                eval_dist += (-r) * active.float()
                active = active & ~done
                cur = action
                if not active.any():
                    break
                    
        print(f"Distance EVAL Greedy : {eval_dist.mean().item():.4f}")
        torch.save(
            {
                "epoch": epoch + 1,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
            },
            os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}.pt"),
        )
        if epoch < decay_epochs:
            scheduler_a.step()
            scheduler_c.step()
        actor.train() # On remet en mode entraînement

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Entraînement VRP par REINFORCE avec baseline")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--capacity", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--transformer", action="store_true", help="Utiliser Transformers ou pas")
    ap.add_argument("--cross", action="store_true", help="Activer la pénalité de croisement")
    ap.add_argument("--output", type=str, default="checkpoints")
    args = ap.parse_args()

    torch.manual_seed(42)

    if args.transformer:
        from model.TransformerActor import TransformerVRPActor
        from model.TransformerCritic import TransformerVRPCritic
        actor  = TransformerVRPActor(128)
        critic = TransformerVRPCritic(128)
    else:
        actor  = VRPActor(D=128)
        critic = VRPCritic(128)

    train(actor, critic, n=args.n, capacity=args.capacity, batch_size=args.batch,
          epochs=args.epochs, want_cross=args.cross,
          OUTPUT_DIR=args.output)
    torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict()}, "vrp_checkpoint.pt")
