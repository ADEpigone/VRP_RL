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


def main():
    p = argparse.ArgumentParser(description="Simple VRP inference viewer")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--capacity", type=int, default=20)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)
    actor = VRPActor(args.embedding_dim).to(device).eval()
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        load_actor(actor, path, device)

    pygame.init()
    w, h = 1920, 1080
    area = (20, 20, w - 260, h - 40)
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("VRP Inference - Simple")
    font, small = pygame.font.SysFont("Consolas", 20), pygame.font.SysFont("Consolas", 16)
    clock = pygame.time.Clock()

    env = VRPEnv(args.n, args.capacity, batch_size=1, device=device)

    def reset_scene(new_points, new_demands):
        env.reset(new_points=new_points, new_demands=new_demands)
        hh, cc = actor.init_hidden(1, actor.D)
        return (hh.to(device), cc.to(device)), [], 0, 0.0, False

    def randomize_demands_in_place():
        raw = torch.randint(1, 10, (env.B, env.n), dtype=torch.float, device=device)
        env.demands[:, 1:] = raw
        env.demands[:, 0] = 0
        # Keep replay consistent with the latest demand profile.
        env.base_demands = torch.cat([torch.zeros(env.B, 1, device=device), raw.clone()], dim=1)

    hidden, route, step_i, total_dist, done = reset_scene(True, True)
    max_steps = args.n * 3
    status = "SPACE step | D demands | N points+demands | R replay | Q quit"

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif e.key == pygame.K_r:
                    hidden, route, step_i, total_dist, done = reset_scene(False, False)
                    status = "Replay same scenario"
                elif e.key == pygame.K_d:
                    randomize_demands_in_place()
                    done = False
                    status = "New demands (route kept)"
                elif e.key == pygame.K_n:
                    hidden, route, step_i, total_dist, done = reset_scene(True, True)
                    status = "New points and demands"
                elif e.key == pygame.K_SPACE and not done:
                    with torch.no_grad():
                        probs, hidden = actor.step(env.static, env.dynamic(), env.cur, hidden, env.get_mask())
                    print(probs)
                    action = Categorical(probs).sample() if args.sample else probs.argmax(1)
                    prev, nxt = int(env.cur.item()), int(action.item())
                    _, _, reward, done_t = env.step(action)
                    d = float((-reward).item())
                    route.append((prev, nxt))
                    step_i += 1
                    total_dist += d

                    policy_done = bool(done_t.item())
                    done = policy_done or step_i >= max_steps
                    if done:
                        status = (
                            f"Done | distance={total_dist:.2f}"
                            if policy_done
                            else f"Max steps reached | distance={total_dist:.2f}"
                        )
                    else:
                        status = f"step {step_i}: {prev}->{nxt} | dist={total_dist:.2f}"

        screen.fill((24, 24, 28))
        pts, demands = env.static[0].cpu(), env.demands[0].cpu()
        for a, b in route:
            pygame.draw.line(screen, (90, 180, 255), to_xy(pts[a], area), to_xy(pts[b], area), 3)
        for i, pnt in enumerate(pts):
            demand = float(demands[i])
            color = (220, 70, 70) if i == 0 else (70, 180, 90) if demand <= 1e-6 else (230, 190, 80)
            x, y = to_xy(pnt, area)
            pygame.draw.circle(screen, color, (x, y), 12 if i == 0 else 10)
            if i == int(env.cur.item()):
                pygame.draw.circle(screen, (245, 245, 245), (x, y), 14, 2)
            txt = "D" if i == 0 else f"{i}:{demand:.0f}"
            screen.blit(small.render(txt, True, (230, 230, 235)), (x + 8, y - 8))

        remain = float(demands[1:].sum().item())
        info = [
            status,
            f"load: {float(env.load.item()):.1f}/{args.capacity}",
            f"remaining demand: {remain:.1f}",
            f"mode: {'sample' if args.sample else 'greedy'}",
        ]
        for i, line in enumerate(info):
            screen.blit((font if i == 0 else small).render(line, True, (235, 235, 240)), (w - 800, 30 + i * 28))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
