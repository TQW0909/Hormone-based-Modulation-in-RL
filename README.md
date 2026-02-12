# Hormone-based Modulation in Reinforcement Learning

A reinforcement learning project that modulates PPO (Proximal Policy Optimization) hyperparameters using a bio-inspired hormone system. Three virtual hormones—**Adrenaline (A)**, **Cortisol (C)**, and **Dopamine (D)**—dynamically adjust entropy coefficient, learning rate, and clip range based on novelty, stress, and learning progress.

## Overview

The hormone system runs as a **callback** during PPO training. After each rollout it:

- **Adrenaline (A)** — Responds to novelty (unfamiliar observations). Higher A → more exploration (higher entropy, slightly looser clip).
- **Cortisol (C)** — Responds to stress (e.g., high TD error, poor value accuracy). Higher C → more caution (lower learning rate, tighter updates).
- **Dopamine (D)** — Responds to learning progress (reward trends, TD improvement). Higher D → more exploitation (lower entropy, higher learning rate).

Hormones interact via an **AdvancedHormoneCoupler** (e.g., cortisol suppresses adrenaline and dopamine; dopamine inhibits cortisol) and use pulse-decay dynamics with optional homeostatic setpoints.

## Project Structure

```
.
├── README.md
├── wandb_utils.py           # Weights & Biases run naming and tags
├── Initial_testing/         # Sanity checks for Gymnasium and SB3
│   ├── gymnasium_test.py
│   └── stable_baseline_test.py
└── PPO/
    ├── PPO_hormones.py      # HormonePPOCallback, novelty detector, coupler, progress tracker
    ├── PPO_hormones_run.py  # Training script: PPO + hormones (e.g. CartPole-v1)
    ├── PPO_baseline_discrete.py
    ├── PPO_baseline_continuous.py
    ├── PPO_hormone_discrete.py
    ├── PPO_hormone_continuous.py
    └── test.py
```

## Requirements

- Python 3.8+
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO)
- [PyTorch](https://pytorch.org/)
- [Weights & Biases](https://wandb.ai/) (optional, for logging)

Install with pip:

```bash
pip install gymnasium stable-baselines3 torch wandb
```

## Quick Start

From the project root, run PPO with hormone modulation (e.g. CartPole):

```bash
cd PPO
python PPO_hormones_run.py
```

This uses the config in `PPO_hormones_run.py` (environment, total timesteps, hormone parameters). For baseline (no hormones), use the baseline scripts, e.g.:

```bash
python PPO_baseline_discrete.py   # e.g. CartPole-v1, LunarLander-v3
python PPO_baseline_continuous.py # continuous action environments
```

## Configuration

Key settings in `PPO_hormones_run.py`:

- **Environment:** `ENV_ID` (e.g. `"CartPole-v1"`)
- **Training:** `TOTAL_TIMESTEPS`, `EVAL_FREQ`, `N_EVAL_EPISODES`
- **Hormone callback:** `warmup_rollouts`, `beta_A/C/D`, `base_A/C/D`, `homeo_A/C/D`, `ent0`, `lr0`, `clip0`, and clamp ranges for entropy, LR, and clip.

Logging goes to **Weights & Biases** (project `hormonal-rl`) and to TensorBoard (`tb/`). Checkpoints and evaluation logs are written to `checkpoints/` and `eval_logs/` (create them if missing).

## Hormone → PPO Mapping

| Hormone   | Effect on PPO |
|----------|----------------|
| Adrenaline | ↑ entropy (exploration), ↑ clip range |
| Cortisol   | ↓ learning rate (caution), stability |
| Dopamine   | ↑ learning rate (confidence), ↓ entropy (exploitation) |

All modulation is centered so that baseline hormone levels (e.g. 0.5) yield the default hyperparameters (`ent0`, `lr0`, `clip0`). Clamping and safety floors keep training stable.

## License

See repository for license information.
