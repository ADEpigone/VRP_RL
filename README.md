# VRP_RL

Implémentation de "Reinforcement Learning for Solving the Vehicle Routing Problem" de Nazari et al (2018).

Contient aussi une implémentation avec Transformers.

Réalisé par Alexandre Ducros et Nabil Hamoudi

## Prérequis

```bash
pip install torch pygame
```

## Quickstart

```bash
python train.py --output checkpoints
python benchmark.py without_cross_VRP10/papier_040.pt transformers_VRP10/trans10_040.pt
python visualize_inference.py --checkpoint without_cross_VRP10/papier_040.pt --checkpoint2 transformers_VRP10/trans10_040.pt
```

## Structure

```
train.py                  # script de training
benchmark.py              # évaluation sur instances
visualize_inference.py    # visualisation pygame
vrp_env.py                # environnement VRP
model/
  VRPActor.py             # acteur du papier
  VRPCritic.py            # critique du papier
  TransformerActor.py     # acteur Transformer
  TransformerCritic.py    # critique Transformer
  attention.py            # mécanisme d'attention (glimpse !)
```

## Entraînement

```bash
python train.py [--n N] [--capacity C] [--batch B] [--epochs E] [--transformer] [--cross] [--output DIR]
```

- `--n` (défaut: 20) : nombre de clients
- `--capacity` (défaut: 30) : capacité du véhicule
- `--batch` (défaut: 512) : taille de batch
- `--epochs` (défaut: 60) : nombre d'époques
- `--transformer` : utiliser TransformerActor/Critic
- `--cross` : activer la pénalité de croisement d'arêtes
- `--output` (défaut: `checkpoints`) : dossier de sauvegarde

```bash
python train.py --n 10 --capacity 20 --epochs 40 --output checkpoints_vrp10
python train.py --transformer --cross --output checkpoints_transformer
```

## Benchmark

```bash
python benchmark.py chemin/checkpoint1.pt [chemin/checkpoint2.pt ...] [--samples N] [--vrp {10,20}]
```

- `--samples` (défaut: 10000) : nombre d'instances de test
- `--vrp` (défaut: 20) : taille du problème (10 ou 20)

```bash
python benchmark.py with_cross_VRP10/papier_cross_040.pt
python benchmark.py with_cross_VRP10/papier_cross_040.pt without_cross_VRP10/papier_040.pt --vrp 10
```

## Visualisation

Script interactif pour voir la route générée étape par étape. Permet de comparer deux checkpoints côte à côte.

```bash
python visualize_inference.py [--checkpoint CKPT] [--checkpoint2 CKPT2] [--n N] [--capacity C] [--sample] [--seed S] [--device cpu|cuda]
```

- `--checkpoint` : checkpoint du modèle gauche (défaut: aléatoire)
- `--checkpoint2` : checkpoint du modèle droit pour comparaison
- `--n` (défaut: 10) : nombre de clients
- `--capacity` (défaut: 20) : capacité du véhicule
- `--sample` : rollout stochastique (défaut: si pas mis la stratégie est greedy)
- `--seed` (défaut 42)

Contrôles :

- `SPACE` : exécuter une étape
- `N` : nouvelle scène (nouveaux points + demandes)
- `Q` / `ESC` : quitter
