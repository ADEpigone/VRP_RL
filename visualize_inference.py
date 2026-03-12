import argparse
import math
from pathlib import Path

import torch
from torch.distributions import Categorical

try:
    import pygame
except ImportError as exc:
    raise SystemExit("Install pygame first: pip install pygame") from exc

from model.VRPActor import VRPActor
from vrp_env import VRPEnv

"""
TERRITOIRE 100% vibe codé

"""

def load_actor(actor, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        state = state.get("actor_state_dict", state.get("state_dict", state))
    state = {k.replace("module.", "").replace("actor.", ""): v for k, v in state.items()}
    actor.load_state_dict(state, strict=True)


def to_xy(point, area):
    x0, y0, w, h = area
    return int(x0 + float(point[0]) * w), int(y0 + float(point[1]) * h)


def exact_vrp_opt(env, capacity, max_exact_n=15):
    """Returns (cost, edges, note). edges = list of (from_idx, to_idx) node-index pairs."""
    points = env.static[0].detach().cpu()
    demands = env.demands[0, 1:].detach().cpu().tolist()
    n_clients = len(demands)

    if n_clients == 0:
        return 0.0, [], None
    if n_clients > max_exact_n:
        return None, None, f"Exact solver disabled (n={n_clients} > {max_exact_n})"
    if any(float(d) > float(capacity) + 1e-9 for d in demands):
        return math.inf, None, "Infeasible: a demand exceeds capacity"

    dmat = torch.cdist(points, points, p=2).tolist()
    full = 1 << n_clients
    inf = float("inf")

    demand_sum = [0.0] * full
    feasible = [False] * full
    feasible[0] = True
    for mask in range(1, full):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_m = mask ^ lsb
        demand_sum[mask] = demand_sum[prev_m] + float(demands[idx])
        feasible[mask] = demand_sum[mask] <= float(capacity) + 1e-9

    tsp_dp = [[inf] * n_clients for _ in range(full)]
    route_cost = [inf] * full
    route_cost[0] = 0.0

    for j in range(n_clients):
        tsp_dp[1 << j][j] = dmat[0][j + 1]

    for mask in range(1, full):
        if mask & (mask - 1):
            mm = mask
            while mm:
                lsb_j = mm & -mm
                j = lsb_j.bit_length() - 1
                prev_mask = mask ^ lsb_j
                best = inf
                pm = prev_mask
                while pm:
                    lsb_i = pm & -pm
                    i = lsb_i.bit_length() - 1
                    v = tsp_dp[prev_mask][i] + dmat[i + 1][j + 1]
                    if v < best:
                        best = v
                    pm ^= lsb_i
                tsp_dp[mask][j] = best
                mm ^= lsb_j

        if feasible[mask]:
            bc = inf
            mm = mask
            while mm:
                lsb_j = mm & -mm
                j = lsb_j.bit_length() - 1
                v = tsp_dp[mask][j] + dmat[j + 1][0]
                if v < bc:
                    bc = v
                mm ^= lsb_j
            route_cost[mask] = bc

    best_part = [inf] * full
    best_part[0] = 0.0
    for mask in range(1, full):
        anchor = mask & -mask
        sub = mask
        while sub:
            if (sub & anchor) and feasible[sub]:
                v = route_cost[sub] + best_part[mask ^ sub]
                if v < best_part[mask]:
                    best_part[mask] = v
            sub = (sub - 1) & mask

    def recon_edges(mask):
        bj, bc = -1, inf
        mm = mask
        while mm:
            lsb_j = mm & -mm
            j = lsb_j.bit_length() - 1
            v = tsp_dp[mask][j] + dmat[j + 1][0]
            if v < bc:
                bc, bj = v, j
            mm ^= lsb_j
        path = [bj]
        cm, cj = mask, bj
        while cm & (cm - 1):
            pm2 = cm ^ (1 << cj)
            bi, bc2 = -1, inf
            pm = pm2
            while pm:
                lsb_i = pm & -pm
                i = lsb_i.bit_length() - 1
                v = tsp_dp[pm2][i] + dmat[i + 1][cj + 1]
                if v < bc2:
                    bc2, bi = v, i
                pm ^= lsb_i
            path.append(bi)
            cm, cj = pm2, bi
        path.reverse()
        route = [0] + [j + 1 for j in path] + [0]
        return [(route[k], route[k + 1]) for k in range(len(route) - 1)]

    edges = []
    rem = full - 1
    while rem:
        anchor = rem & -rem
        sub = rem
        while sub:
            if (sub & anchor) and feasible[sub]:
                if abs(route_cost[sub] + best_part[rem ^ sub] - best_part[rem]) < 1e-9:
                    edges.extend(recon_edges(sub))
                    rem ^= sub
                    break
            sub = (sub - 1) & rem

    return best_part[full - 1], edges, None


def _make_model_panel(actor, label, device, opt_cost, opt_note):
    hh, cc = actor.init_hidden(1, actor.D)
    return {
        "actor": actor,
        "hidden": (hh.to(device), cc.to(device)),
        "edges": [],
        "step_i": 0,
        "total_dist": 0.0,
        "done": False,
        "label": label,
        "opt_cost": opt_cost,
        "opt_note": opt_note,
        "status": "ready",
    }


def _make_optimal_panel(label, opt_cost, opt_edges, opt_note):
    dist = opt_cost if (opt_cost is not None and math.isfinite(opt_cost)) else 0.0
    if opt_cost is not None and math.isfinite(opt_cost):
        status = f"dist: {opt_cost:.3f}"
    else:
        status = f"n/a: {opt_note or ''}"
    return {
        "actor": None,
        "hidden": None,
        "edges": opt_edges or [],
        "step_i": 0,
        "total_dist": dist,
        "done": True,
        "label": label,
        "opt_cost": opt_cost,
        "opt_note": opt_note,
        "status": status,
    }


def draw_panel(screen, pts, demands, panel, cur_node, area, load, remaining, capacity, small_f, font_f):
    x0, y0, pw, ph = area
    pygame.draw.rect(screen, (45, 45, 55), (x0 - 1, y0 - 1, pw + 2, ph + 2), 1)

    is_opt = panel["actor"] is None
    edge_color = (100, 220, 130) if is_opt else (90, 180, 255)
    for a, b in panel["edges"]:
        pygame.draw.line(screen, edge_color, to_xy(pts[a], area), to_xy(pts[b], area), 3)

    for i, pnt in enumerate(pts):
        demand = float(demands[i])
        color = (220, 70, 70) if i == 0 else (70, 180, 90) if demand <= 1e-6 else (230, 190, 80)
        x, y = to_xy(pnt, area)
        pygame.draw.circle(screen, color, (x, y), 12 if i == 0 else 10)
        if i == cur_node and not is_opt:
            pygame.draw.circle(screen, (245, 245, 245), (x, y), 14, 2)
        txt = "D" if i == 0 else f"{i}:{demand:.0f}"
        screen.blit(small_f.render(txt, True, (230, 230, 235)), (x + 8, y - 8))

    lbl_color = (140, 255, 180) if is_opt else (140, 200, 255)
    screen.blit(font_f.render(panel["label"], True, lbl_color), (x0, y0 - 32))

    opt_cost = panel["opt_cost"]
    has_opt = (opt_cost is not None) and math.isfinite(opt_cost) and opt_cost > 1e-9
    opt_ratio = (panel["total_dist"] / opt_cost) if (has_opt and panel["total_dist"] > 0) else 0.0

    lines = [panel["status"]]
    if not is_opt:
        lines.append(f"load: {load:.1f}/{capacity}  rem: {remaining:.1f}")
        lines.append(f"dist: {panel['total_dist']:.3f}")
        if has_opt:
            lines.append(f"dist/opt: {opt_ratio:.3f}x")

    for i2, line in enumerate(lines):
        screen.blit(small_f.render(line, True, (235, 235, 240)), (x0, y0 + ph + 6 + i2 * 20))


def main():
    p = argparse.ArgumentParser(description="VRP inference comparison viewer")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--checkpoint2", type=str, default=None)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--capacity", type=int, default=20)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)

    use_optimal_right = args.checkpoint2 is None

    actor_a = VRPActor(args.embedding_dim).to(device).eval()
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        load_actor(actor_a, path, device)
    label_a = Path(args.checkpoint).name if args.checkpoint else "model (no ckpt)"

    actor_b = None
    label_b = "OPTIMAL"
    if not use_optimal_right:
        actor_b = VRPActor(args.embedding_dim).to(device).eval()
        path2 = Path(args.checkpoint2)
        if not path2.exists():
            raise SystemExit(f"Checkpoint not found: {path2}")
        load_actor(actor_b, path2, device)
        label_b = Path(args.checkpoint2).name

    pygame.init()
    W, H = 1920, 1080
    MARG = 12
    panel_w = (W - 3 * MARG) // 2
    panel_h = H - 210
    area_l = (MARG, 80, panel_w, panel_h)
    area_r = (2 * MARG + panel_w, 80, panel_w, panel_h)

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("VRP Comparison")
    font = pygame.font.SysFont("Consolas", 20)
    small = pygame.font.SysFont("Consolas", 16)
    clock = pygame.time.Clock()

    env_a = VRPEnv(args.n, args.capacity, batch_size=1, device=device)
    env_b = VRPEnv(args.n, args.capacity, batch_size=1, device=device) if not use_optimal_right else None

    def reset_scene(new_points, new_demands):
        env_a.reset(new_points=new_points, new_demands=new_demands)
        if env_b is not None:
            env_b.base_static = env_a.base_static.clone()
            env_b.base_demands = env_a.base_demands.clone()
            env_b.reset(new_points=False, new_demands=False)
        opt_cost, opt_edges, opt_note = exact_vrp_opt(env_a, args.capacity)
        pa = _make_model_panel(actor_a, label_a, device, opt_cost, opt_note)
        pb = (_make_optimal_panel(label_b, opt_cost, opt_edges, opt_note) if use_optimal_right
              else _make_model_panel(actor_b, label_b, device, opt_cost, opt_note))
        return pa, pb

    def step_model(panel, env):
        if panel["done"] or panel["actor"] is None:
            return
        with torch.no_grad():
            probs, new_hidden = panel["actor"].step(
                env.static, env.dynamic(), env.cur, panel["hidden"], env.get_mask()
            )
        action = Categorical(probs).sample() if args.sample else probs.argmax(1)
        prev, nxt = int(env.cur.item()), int(action.item())
        _, _, reward, done_t = env.step(action)
        panel["edges"].append((prev, nxt))
        panel["step_i"] += 1
        panel["total_dist"] += float((-reward).item())
        panel["hidden"] = new_hidden
        policy_done = bool(done_t.item())
        panel["done"] = policy_done or panel["step_i"] >= args.n * 3
        if panel["done"]:
            panel["status"] = (f"Done  dist={panel['total_dist']:.3f}" if policy_done
                               else f"MaxSt dist={panel['total_dist']:.3f}")
        else:
            panel["status"] = f"step {panel['step_i']}: {prev}->{nxt}"

    panel_a, panel_b = reset_scene(True, True)
    controls = "SPACE:step | N:new scene | R:replay | Q:quit"

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_n:
                    panel_a, panel_b = reset_scene(True, True)
                elif e.key == pygame.K_r:
                    panel_a, panel_b = reset_scene(False, False)
                elif e.key == pygame.K_SPACE:
                    step_model(panel_a, env_a)
                    if not use_optimal_right:
                        step_model(panel_b, env_b)

        screen.fill((24, 24, 28))
        pts = env_a.static[0].cpu()
        demands_a = env_a.demands[0].cpu()
        demands_b = env_b.demands[0].cpu() if env_b is not None else demands_a

        load_a = float(env_a.load.item())
        rem_a = float(env_a.demands[0, 1:].sum().item())
        load_b = float(env_b.load.item()) if env_b is not None else 0.0
        rem_b = float(env_b.demands[0, 1:].sum().item()) if env_b is not None else 0.0

        draw_panel(screen, pts, demands_a, panel_a, int(env_a.cur.item()),
                   area_l, load_a, rem_a, args.capacity, small, font)
        cur_b = int(env_b.cur.item()) if env_b is not None else None
        draw_panel(screen, pts, demands_b, panel_b, cur_b,
                   area_r, load_b, rem_b, args.capacity, small, font)

        pygame.draw.line(screen, (60, 60, 70), (W // 2, 10), (W // 2, H - 10), 1)

        info = f"n={args.n}  cap={args.capacity}  {'sample' if args.sample else 'greedy'}  seed={args.seed}"
        screen.blit(small.render(info, True, (160, 160, 180)), (W // 2 - 180, 12))
        screen.blit(small.render(controls, True, (180, 180, 200)), (MARG, H - 28))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
