import argparse
from pathlib import Path

import torch
from torch.distributions import Categorical

try:
    import pygame
except ImportError as exc:
    raise SystemExit("Installez pygame d'abord : pip install pygame") from exc

from model.VRPActor import VRPActor
from vrp_env import VRPEnv

"""
Visualisation générée par Sonnet 4.6. Mais commentée et relue à la main.
Le script permet l'inférence de deux checkpoints et de les comparer.
Permet aussi d'ajouter et de créer des instances à la main. -> contrôles dans le readme
"""

def load_actor(actor, path, device):
    """
    Helper pour load les modèles
    Il prend un acteur déjà init et lui load ses poids
    """
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        state = state.get("actor_state_dict", state.get("state_dict", state))
    state = {k.replace("module.", "").replace("actor.", ""): v for k, v in state.items()}
    actor.load_state_dict(state, strict=True)


def make_actor(checkpoint_path, embedding_dim, device):
    """
    Helper pour créer et charger un acteur depuis un checkpoint
    Détecte automatiquement si c'est un Transformer ou un VRPActor classique
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise SystemExit(f"Checkpoint introuvable : {path}")
    if "transformers" in str(path):
        from model.TransformerActor import TransformerVRPActor
        actor = TransformerVRPActor(D=embedding_dim).to(device).eval()
    else:
        actor = VRPActor(embedding_dim).to(device).eval()
    load_actor(actor, path, device)
    return actor


def to_xy(point, area):
    """
    Helper pour convertir des coordonnées normalisées (0-1) en coordonnées d'écran
    """
    x0, y0, w, h = area
    return int(x0 + float(point[0]) * w), int(y0 + float(point[1]) * h)


def _make_model_panel(actor, label, device):
    """
    Helper pour créer un panneau de modèle pour la visualisation
    """
    hh, cc = actor.init_hidden(1, actor.D)
    return {
        "actor": actor,
        "hidden": (hh.to(device), cc.to(device)),
        "edges": [],
        "step_i": 0,
        "total_dist": 0.0,
        "done": False,
        "label": label,
        "status": "prêt",
    }



def draw_panel(screen, pts, demands, panel, cur_node, area, load, remaining, capacity, small_f, font_f):
    """
    Helper pour dessiner un panneau de modèle pour la visualisation
    """
    x0, y0, pw, ph = area
    pygame.draw.rect(screen, (45, 45, 55), (x0 - 1, y0 - 1, pw + 2, ph + 2), 1)

    # Si n'a pas de modèle, initialement on mettait l'opti
    # reste historique pour dire qu'on affiche "rien"
    is_opt = panel["actor"] is None

    # On affiche les arêtes AVANT les points pour qu'elles soient dessous
    edge_color = (100, 220, 130) if is_opt else (90, 180, 255)
    for a, b in panel["edges"]:
        pygame.draw.line(screen, edge_color, to_xy(pts[a], area), to_xy(pts[b], area), 3)

    # On affiche les points par dessus les arêtes
    # style spécial pour les dépôts
    for i, pnt in enumerate(pts):
        demand = float(demands[i])
        color = (220, 70, 70) if i == 0 else (70, 180, 90) if demand <= 1e-6 else (230, 190, 80)
        x, y = to_xy(pnt, area)
        pygame.draw.circle(screen, color, (x, y), 12 if i == 0 else 10)
        if i == cur_node and not is_opt:
            pygame.draw.circle(screen, (245, 245, 245), (x, y), 14, 2)
        txt = "D" if i == 0 else f"{i}:{demand:.0f}"
        screen.blit(small_f.render(txt, True, (230, 230, 235)), (x + 8, y - 8))

    # Label du pannel / modèle
    lbl_color = (140, 255, 180) if is_opt else (140, 200, 255)
    screen.blit(font_f.render(panel["label"], True, lbl_color), (x0, y0 - 32))

    # Affichage du status
    lines = [panel["status"]]
    lines.append(f"charge: {load:.1f}/{capacity}  rest.: {remaining:.1f}")
    lines.append(f"dist : {panel['total_dist']:.3f}")

    # Affichage des infos de status en bas du panneau
    for i2, line in enumerate(lines):
        screen.blit(small_f.render(line, True, (235, 235, 240)), (x0, y0 + ph + 6 + i2 * 20))


def main():
    p = argparse.ArgumentParser(description="Visualiseur de comparaison d'inférence VRP")
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
    
    # On load les deux modèles (ou un seul)
    actor_a = VRPActor(args.embedding_dim).to(device).eval()
    if args.checkpoint:
        print(args.checkpoint)
        actor_a = make_actor(args.checkpoint, args.embedding_dim, device)
    label_a = Path(args.checkpoint).name if args.checkpoint else "modèle (sans checkpoint)"

    actor_b = None
    label_b = ""
    if not use_optimal_right:
        print(args.checkpoint2)
        actor_b = make_actor(args.checkpoint2, args.embedding_dim, device)
        label_b = Path(args.checkpoint2).name

    # Init de Pygame et création de la fenêtre
    pygame.init()
    W, H = 1920, 1080
    MARG = 12
    panel_w = (W - 3 * MARG) // 2
    panel_h = H - 210
    area_l = (MARG, 80, panel_w, panel_h)
    area_r = (2 * MARG + panel_w, 80, panel_w, panel_h)

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Comparaison VRP")
    font = pygame.font.SysFont("Consolas", 20)
    small = pygame.font.SysFont("Consolas", 16)
    clock = pygame.time.Clock()

    # Init des env et des panneaux de contrôle

    env_a = VRPEnv(args.n, args.capacity, batch_size=1, device=device)
    env_b = VRPEnv(args.n, args.capacity, batch_size=1, device=device) if not use_optimal_right else None


    # Les fonctions suivantes sont définies ici pour des question de scope
    def make_panels():
        # Crée les deux panneaux acteur, panel_b est None si on n'a pas de second modèle
        pa = _make_model_panel(actor_a, label_a, device)
        pb = None if use_optimal_right else _make_model_panel(actor_b, label_b, device)
        return pa, pb

    def reset_scene():
        env_a.reset()
        if env_b is not None:
            env_b.base_static = env_a.base_static.clone()
            env_b.base_demands = env_a.base_demands.clone()
            env_b.reset(new_points=False, new_demands=False)
        return make_panels()

    def step_model(panel, env):
        """
        Fonction qui effectue une step
        Dans la logique de celle de train.py
        """
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
        panel["done"] = policy_done or panel["step_i"] >= env.n * 3
        if panel["done"]:
            panel["status"] = (f"Terminé  dist={panel['total_dist']:.3f}" if policy_done
                               else f"MaxÉtapes dist={panel['total_dist']:.3f}")
        else:
            panel["status"] = f"étape {panel['step_i']} : {prev}->{nxt}"

    panel_a, panel_b = reset_scene()
    controls_auto   = "ESPACE:avancer | N:nouveau | M:manuel | Q:quitter"
    controls_manual = "ClicG:placer nœud | Entrée:confirmer | Suppr:annuler | Échap:annuler"

    manual_mode    = False
    manual_pts     = [] 
    manual_demands = []  

    def enter_manual():
        """
        Reset la scène et entre en mode manuel pour placer les points à la main
        """
        nonlocal manual_mode, manual_pts, manual_demands, panel_a, panel_b
        manual_mode = True
        manual_pts, manual_demands = [], []
        panel_a["edges"] = []
        panel_a["status"] = "prêt"
        if panel_b is not None:
            panel_b["edges"] = []
            panel_b["status"] = "prêt"

    def cancel_manual():
        """
        Annule le mode manuel et réinitialise les points
        """
        nonlocal manual_mode, manual_pts, manual_demands
        manual_mode = False
        manual_pts, manual_demands = [], []

    def finalize_manual():
        """
        Conversion de ce qui vient d'être placé en env et panneaux, puis exit du mode manuel
        """ 
        nonlocal manual_mode, manual_pts, manual_demands, panel_a, panel_b, env_a, env_b
        if len(manual_pts) < 2:
            return
        manual_mode = False
        n_cust = len(manual_pts) - 1
        pts_t = torch.tensor(manual_pts, dtype=torch.float, device=device).unsqueeze(0)
        dem_t = torch.tensor([0.0] + manual_demands, dtype=torch.float, device=device).unsqueeze(0)
        env_a = VRPEnv(n_cust, args.capacity, batch_size=1, device=device)
        env_a.base_static  = pts_t
        env_a.base_demands = dem_t
        env_a.reset(new_points=False, new_demands=False)

        # Si l'on doit mirror ou pas ce que l'on vient de faire
        if not use_optimal_right:
            env_b = VRPEnv(n_cust, args.capacity, batch_size=1, device=device)
            env_b.base_static  = pts_t.clone()
            env_b.base_demands = dem_t.clone()
            env_b.reset(new_points=False, new_demands=False)


        panel_a, panel_b = make_panels()
        manual_pts, manual_demands = [], []

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and manual_mode:
                # Clic en coordonnées si dans mode de génération d'instance à la main
                mx, my = e.pos
                x0, y0, pw, ph = area_l
                nx = (mx - x0) / pw
                ny = (my - y0) / ph
                if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                    manual_pts.append([nx, ny])
                    if len(manual_pts) > 1:
                        manual_demands.append(int(torch.randint(1, 10, (1,)).item()))
            elif e.type == pygame.KEYDOWN:
                # Méthodes pour quitter
                if e.key == pygame.K_q:
                    running = False
                elif e.key == pygame.K_ESCAPE:
                    if manual_mode:
                        cancel_manual()
                    else:
                        running = False
                # Pour lancer le mode manuel
                elif e.key == pygame.K_m:
                    enter_manual()
                # Pour confirmer la création manuelle d'instance
                elif e.key == pygame.K_RETURN and manual_mode:
                    finalize_manual()
                # Pour annuler le dernier point placé
                elif e.key == pygame.K_BACKSPACE and manual_mode:
                    if manual_pts:
                        manual_pts.pop()
                        if manual_demands:
                            manual_demands.pop()
                # Pour reset la scène et en générer une nouvelle
                elif e.key == pygame.K_n and not manual_mode:
                    panel_a, panel_b = reset_scene()
                # Pour faire une step d'inférence
                elif e.key == pygame.K_SPACE and not manual_mode:
                    step_model(panel_a, env_a)
                    if not use_optimal_right:
                        step_model(panel_b, env_b)

        screen.fill((24, 24, 28))
        pts = env_a.static[0].cpu()
        
        # Calcul de stats à afficher
        demands_a = env_a.demands[0].cpu()
        demands_b = env_b.demands[0].cpu() if env_b is not None else demands_a

        load_a = float(env_a.load.item())
        rem_a = float(env_a.demands[0, 1:].sum().item())
        load_b = float(env_b.load.item()) if env_b is not None else 0.0
        rem_b = float(env_b.demands[0, 1:].sum().item()) if env_b is not None else 0.0

        if not manual_mode:
            # Affichage des stats si on le doit
            draw_panel(screen, pts, demands_a, panel_a, int(env_a.cur.item()),
                       area_l, load_a, rem_a, args.capacity, small, font)
            cur_b = int(env_b.cur.item()) if env_b is not None else None
            if panel_b is not None:
                draw_panel(screen, pts, demands_b, panel_b, cur_b,
                           area_r, load_b, rem_b, args.capacity, small, font)

        pygame.draw.line(screen, (60, 60, 70), (W // 2, 10), (W // 2, H - 10), 1)

        if manual_mode:
            n_placed = len(manual_pts)
            if n_placed == 0:
                hint = "Cliquez pour placer le DÉPÔT"
            elif n_placed == 1:
                hint = "Dépôt placé, cliquez pour ajouter des clients"
            else:
                hint = f"{n_placed - 1} client(s) — Entrée pour confirmer, Suppr pour annuler"
            screen.blit(font.render("-- MODE MANUEL --", True, (255, 220, 60)), (W // 2 - 100, 46))
            screen.blit(small.render(hint, True, (220, 200, 80)), (MARG, H - 50))
            for area in (area_l, area_r):
                for i, pt in enumerate(manual_pts):
                    x, y = to_xy(pt, area)
                    col = (220, 70, 70) if i == 0 else (230, 190, 80)
                    pygame.draw.circle(screen, col, (x, y), 12 if i == 0 else 10)
                    lbl = "D" if i == 0 else f"{i}:{manual_demands[i - 1]}"
                    screen.blit(small.render(lbl, True, (230, 230, 235)), (x + 8, y - 8))

        cur_n = env_a.n
        info = f"n={cur_n}  cap={args.capacity}  {'aléatoire' if args.sample else 'glouton'}  graine={args.seed}"
        screen.blit(small.render(info, True, (160, 160, 180)), (W // 2 - 180, 12))
        controls = controls_manual if manual_mode else controls_auto
        screen.blit(small.render(controls, True, (180, 180, 200)), (MARG, H - 28))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
