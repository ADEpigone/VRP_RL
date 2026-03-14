
import argparse
import time
from pathlib import Path

import torch

from model.VRPActor import VRPActor
from model.TransformerActor import TransformerVRPActor
from vrp_env import VRPEnv

def rollout(actor, static_all, demands_all, capacity, device, batch_size):
    """
    Un rollout pour un acteur en fonction d'instances
    """
    N_total = static_all.shape[0]
    N = static_all.shape[1] - 1
    max_steps = N * 4

    all_dists = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for start in range(0, N_total, batch_size):
        B = min(batch_size, N_total - start)
        static = static_all[start:start + B].to(device)
        demands = demands_all[start:start + B].to(device)

        env = VRPEnv(N, capacity, batch_size=B, device=device)
        env.base_static = static.clone()
        env.base_demands = demands.clone()
        env.reset(new_points=False, new_demands=False)

        h, c = actor.init_hidden(B, actor.D)
        h, c = h.to(device), c.to(device)

        done = torch.zeros(B, dtype=torch.bool, device=device)
        dist = torch.zeros(B, device=device)

        for step in range(max_steps):
            if done.all():
                break
            with torch.no_grad():
                probs, (h, c) = actor.step(
                    env.static, env.dynamic(), env.cur, (h, c), env.get_mask()
                )
            action = probs.argmax(1)
            _, _, reward, done_step = env.step(action)
            dist += (-reward) * (~done).float()
            done |= done_step

        all_dists.extend(dist.cpu().tolist())
        done_so_far = start + B
        print(f"  [{done_so_far:>{len(str(N_total))}}/{N_total}  {100*done_so_far/N_total:5.1f}%]",
              end="\r", flush=True)

    print(" " * 50, end="\r")

    vram = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == "cuda" else 0.0
    return all_dists, vram


def bench(actor, name, static_all, demands_all, capacity, device, batch_size):
    """
    Prends un acteur et le bench sur les instances données, retourne un dico contenant les résultats
    """
    print(f"  Benchmark de {name} ...", flush=True)
    t0 = time.perf_counter()
    dists, vram = rollout(actor, static_all, demands_all, capacity, device, batch_size)
    elapsed = time.perf_counter() - t0

    n = len(dists)
    mu = sum(dists) / n

    return dict(
        name=name,
        n=n,
        mean_dist=mu,
        min_dist=min(dists),
        max_dist=max(dists),
        time_s=elapsed,
        vram_mb=vram,
    )


def print_table(results):
    metrics = [
        ("Echant.", lambda r: f"{r['n']:,}"),
        ("Dist moy.", lambda r: f"{r['mean_dist']:.4f}"),
        ("Dist min", lambda r: f"{r['min_dist']:.4f}"),
        ("Dist max", lambda r: f"{r['max_dist']:.4f}"),
        ("Temps (s)", lambda r: f"{r['time_s']:.2f}"),
        ("RAM/VRAM utilisée", lambda r: f"{r['vram_mb']:.1f}"),
    ]

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
    static_all = torch.rand(args.samples, n + 1, 2)
    raw = torch.randint(1, 10, (args.samples, n), dtype=torch.float)
    demands_all = torch.cat([torch.zeros(args.samples, 1), raw], dim=1)

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
    for actor, name in models:
        r = bench(actor, name, static_all, demands_all, capacity, device, batch_size)
        results.append(r)
        print(f"  -> Dist moy={r['mean_dist']:.4f}  Temps={r['time_s']:.1f}s\n")

    print_table(results)
