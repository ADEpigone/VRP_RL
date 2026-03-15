
import argparse
import time
from pathlib import Path

import torch

from model.VRPActor import VRPActor
from model.TransformerActor import TransformerVRPActor
from vrp_env import VRPEnv

def rollout(actor, static_all, demands_all, capacity, device, batch_size, frac_initial=1.0):
    """
    Rollout : classique (frac_initial=1.0) ou dynamique.
    En mode dynamique, (1-frac_initial) des clients arrivent en online,
    révélés uniformément sur le premier tiers du temps aloué (alloué ?).
    """
    N_total = static_all.shape[0]
    N = static_all.shape[1] - 1
    is_dynamic = frac_initial < 1.0

    # En dynamique on alloue plus de steps pour laisser le temps aux clients d'arriver
    # Même si ne devrait rien changer
    # -> sécurité
    if is_dynamic:
        max_steps = N * 6
    else:
        max_steps = N * 4

    # Le reveal se fait sur la première moitié des steps
    horizon = max_steps // 2
    n_init = max(1, int(N * frac_initial))
    n_late = N - n_init

    # Pour mesurer la VRAM de pointe sur GPU
    # marche pas pour RAM mais....
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    all_dists = []

    for start in range(0, N_total, batch_size):
        B = min(batch_size, N_total - start)
        static   = static_all[start:start + B].to(device)
        true_dem = demands_all[start:start + B].clone().to(device)

        # hidden = les demandes pas encore révélées (masquées à l'agent)
        hidden = torch.zeros(B, N + 1, device=device)
        arrival_step = torch.zeros(B, N, dtype=torch.long, device=device)
        if n_late > 0:
            # Ordre aléatoire des clients
            reveal_order = torch.stack([torch.randperm(N, device=device) for _ in range(B)])
            # Clients qu'on va real plus tard
            # Ensuite on leur donne leur step d'arrivée
            # Et on en fait un masque !
            late_cols = reveal_order[:, n_init:]
            steps = (torch.arange(n_late, device=device).float() * (horizon - 1) / max(n_late - 1, 1)).long() + 1 
            arrival_step.scatter_(1, late_cols, steps.unsqueeze(0).expand(B, -1)) 
            hidden[:, 1:] = true_dem[:, 1:] * (arrival_step > 0).float()

        env = VRPEnv(N, capacity, batch_size=B, device=device)
        env.base_static  = static.clone()
        # L'env ne voit que les demandes déjà révélées au départ
        # on ajoute à la main (de manière pas très propre) les demandes qui arrivent
        env.base_demands = (true_dem - hidden).clone()
        env.reset(new_points=False, new_demands=False)

        # Init du RNN
        h, c = actor.init_hidden(B, actor.D)
        h, c = h.to(device), c.to(device)

        done = torch.zeros(B, dtype=torch.bool, device=device)
        dist = torch.zeros(B, device=device)

        for step in range(max_steps):
            # Logique de reveal
            # On récupère ceux qui devaient arriver, on les injecte et enlève du masque
            if n_late > 0 and 0 < step <= horizon:
                arrive_mask = (arrival_step == step) 
                to_add = hidden[:, 1:] * arrive_mask.float() 
                env.demands[:, 1:] += to_add
                hidden[:, 1:] -= to_add 
            
            # Plus rien à hide, reveal ou satisfaire ?
            done = (hidden[:, 1:].sum(1) < 1e-4) & (env.demands[:, 1:].sum(1) < 1e-4) & (env.cur == 0)  
            if done.all():
                break

            with torch.no_grad():
                probs, (h, c) = actor.step(env.static, env.dynamic(), env.cur, (h, c), env.get_mask())
            action = probs.argmax(1)
            # Ici on ne prend pas en compte le done
            # A cause du rollout dynamique qui est une "couche par dessus"
            # Il aurait p-ê fallu ajouter cette couche dan sl'env...
            _, _, reward, _ = env.step(action)
            dist += (-reward) * (~done).float()

        all_dists.extend(dist.cpu().tolist())
        done_so_far = start + B
        # Barre de prog, pas besoin de tqdm donc \r
        print(f"  [{done_so_far:>{len(str(N_total))}}/{N_total}  {100*done_so_far/N_total:5.1f}%]",
              end="\r", flush=True)

    print(" " * 50, end="\r")
    vram = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
    return all_dists, vram


def bench(actor, name, static_all, demands_all, capacity, device, batch_size, frac_initial=1.0):
    """
    Bench un modèle sur un ensemble d'instances données  
    """
    label = name if frac_initial >= 1.0 else f"{name} (dyn {int(frac_initial * 100)}%)"
    print(f"  Benchmark de {label} ...", flush=True)
    t0 = time.perf_counter()
    dists, vram = rollout(actor, static_all, demands_all, capacity, device, batch_size, frac_initial)
    elapsed = time.perf_counter() - t0
    n = len(dists)
    mu = sum(dists) / n
    return dict(name=label, n=n, mean_dist=mu, min_dist=min(dists), max_dist=max(dists), time_s=elapsed, vram_mb=vram)


def print_table(results):
    """
    Fonction d'affichage de la table
    """
    metrics = [
        ("Echant.", lambda r: f"{r['n']:,}"),
        ("Dist moy.", lambda r: f"{r['mean_dist']:.4f}"),
        ("Dist min", lambda r: f"{r['min_dist']:.4f}"),
        ("Dist max", lambda r: f"{r['max_dist']:.4f}"),
        ("Temps (s)", lambda r: f"{r['time_s']:.2f}"),
        ("RAM/VRAM utilisée", lambda r: f"{r['vram_mb']:.1f}"),
    ]

    # Calcul des largeurs de colonnes : on prend le max entre le header et les valeurs
    lw = max(len(m[0]) for m in metrics)
    cw = max(len(r["name"]) for r in results)
    for _, fn in metrics:
        for r in results:
            cw = max(cw, len(fn(r)))
    cw += 2

    sep    = "  " + "-" * lw + "  " + "  ".join("-" * cw for _ in results)
    header = "  " + " " * lw + "  " + "  ".join(f"{r['name']:^{cw}}" for r in results)

    print()
    print(header)
    print(sep)
    for label, fn in metrics:
        line = f"  {label:<{lw}}  " + "  ".join(f"{fn(r):>{cw}}" for r in results)
        print(line)
    print(sep)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", metavar="check")
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--vrp", type=int, choices=[10, 20], default=20)
    ap.add_argument("--dynamic", action="store_true", help="Inclure un benchmark dynamique (demandes online)")
    ap.add_argument("--frac_initial", type=float, default=0.5,
                    help="Fraction des clients connus au départ en mode dynamique (défaut: 0.5)")
    args = ap.parse_args()

    embedding_dim = 128
    if args.vrp == 10:
        n, capacity = 10, 20
    else:
        n, capacity = 20, 30
    batch_size = 512
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nGénération de {args.samples:,} instances (n={n}, cap={capacity}, seed={seed}) ...")
    torch.manual_seed(seed)
    # static : coordonnées aléatoires dans [0,1]^2
    static_all = torch.rand(args.samples, n + 1, 2)
    # demandes entières selon papier
    raw = torch.randint(1, 10, (args.samples, n), dtype=torch.float)
    demands_all = torch.cat([torch.zeros(args.samples, 1), raw], dim=1)

    # Chargement des checkpoints passés en argument
    models = []
    for path in args.checkpoints:
        label = Path(path).parent.name
        is_transformer = "transformer" in path.lower()
        actor = TransformerVRPActor(embedding_dim) if is_transformer else VRPActor(embedding_dim)
        actor = actor.to(device).eval()
        state = torch.load(path, map_location=device, weights_only=False)
        state = state.get("actor_state_dict", state.get("state_dict", state))
        state = {k.replace("module.", "").replace("actor.", ""): v for k, v in state.items()}
        actor.load_state_dict(state, strict=True)
        models.append((actor, label))

    print(f"Device: {device}  |  batch_size={batch_size}\n")
    results = []

    if not args.dynamic:
        args.frac_initial = 1.0
    
    if args.dynamic:
        print(f"Benchmark dynamique (frac_initial={args.frac_initial}) :\n")
    for actor, name in models:
        r = bench(actor, name, static_all, demands_all, capacity, device, batch_size, args.frac_initial)
        results.append(r)
        print(f"  -> Dist moy={r['mean_dist']:.4f}  Temps={r['time_s']:.1f}s\n")

    print_table(results)
