import argparse
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


def _make_model_panel(actor, label, device):
    hh, cc = actor.init_hidden(1, actor.D)
    return {
        "actor": actor,
        "hidden": (hh.to(device), cc.to(device)),
        "edges": [],
        "step_i": 0,
        "total_dist": 0.0,
        "done": False,
        "label": label,
        "status": "ready",
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

    lines = [panel["status"]]
    lines.append(f"load: {load:.1f}/{capacity}  rem: {remaining:.1f}")
    lines.append(f"dist: {panel['total_dist']:.3f}")

    for i2, line in enumerate(lines):
        screen.blit(small_f.render(line, True, (235, 235, 240)), (x0, y0 + ph + 6 + i2 * 20))


def main():
    p = argparse.ArgumentParser(description="VRP inference comparison viewer")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--checkpoint2", type=str, default=None)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--capacity", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    args.embedding_dim = 128

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)

    use_optimal_right = args.checkpoint2 is None
    
    actor_a = VRPActor(args.embedding_dim).to(device).eval()
    if args.checkpoint:
        print(args.checkpoint)
        if args.checkpoint.startswith("./transformers/"):
            from model.TransformerActor import TransformerVRPActor
            actor_a = TransformerVRPActor(embed_dim=args.embedding_dim).to(device).eval()
        path = Path(args.checkpoint)
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        load_actor(actor_a, path, device)
    label_a = Path(args.checkpoint).name if args.checkpoint else "model (no ckpt)"

    actor_b = None
    label_b = ""
    if not use_optimal_right:
        print(args.checkpoint2)
        if "transformers" in args.checkpoint2:
            from model.TransformerActor import TransformerVRPActor
            actor_b = TransformerVRPActor(embed_dim=args.embedding_dim).to(device).eval()
        else:
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

    def reset_scene():
        env_a.reset()
        if env_b is not None:
            env_b.base_static = env_a.base_static.clone()
            env_b.base_demands = env_a.base_demands.clone()
            env_b.reset(new_points=False, new_demands=False)
        pa = _make_model_panel(actor_a, label_a, device)
        pb = None if use_optimal_right else _make_model_panel(actor_b, label_b, device)
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

    panel_a, panel_b = reset_scene()
    controls = "SPACE:step | N:new scene | Q:quit"

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_n:
                    panel_a, panel_b = reset_scene()
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
        if panel_b is not None:
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
