# KUKA Mini Stacker

A modern reinforcement learning environment for robot arm control using PyBullet and Gymnasium.

![KUKA Reaching Demo](demo.gif)

## Overview

This repository implements a **4-DOF KUKA IIWA robot arm with gripper** in a reaching task environment, featuring:

- **Modern RL Stack**: Gymnasium API (no deprecated `gym`) + PyBullet physics (no legacy `pybullet_envs`)
- **Multiple Training Methods**: Pure PPO, behavioral cloning (BC) pretraining, or BC + PPO hybrid
- **IK Teacher Policy**: Inverse kinematics-based expert for demonstration collection
- **Docker Support**: Fully containerized with headless rendering for cloud/CI training
- **Comprehensive Logging**: TensorBoard, Weights & Biases, video recording, checkpointing

## Project Status

| Component | Status |
|-----------|--------|
| Environment (`KukaPickPlaceEnv`) | Complete |
| PPO Training | Complete |
| IK Teacher Policy | Complete |
| Behavioral Cloning | Complete |
| BC + PPO Pipeline | Complete |
| Docker Support | Complete |
| Sanity Checks | Complete |

## Quick Start

### Local Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the simulation (GUI mode)
python env_test.py

# Run headless
PYBULLET_RENDER_MODE=headless python env_test.py
```

### Docker

```bash
# Build the image
docker build -t kuka-mini-stacker .

# Run sanity check
docker run --rm kuka-mini-stacker python sanity_check.py

# Run the simulation
docker run --rm kuka-mini-stacker

# Train with PPO
docker run --rm -v $(pwd)/logs:/app/logs kuka-mini-stacker python train_ppo.py --timesteps 100000
```

## Environment

The `KukaPickPlaceEnv` is a Gymnasium-compatible environment featuring:

- **Robot**: Truncated 4-DOF KUKA IIWA arm with prismatic gripper
- **Task**: Reach a randomly placed target on a table
- **Observation** (27-D): `[sin(q), cos(q), qdot, grip_pos, grip_vel, relative_target, ee_pos, prev_action]`
- **Action** (5-D): `[4 arm joint deltas, 1 gripper action]`
- **Reward**: Distance-based shaping with progress bonus and success reward (+50 at distance < 0.05)

```python
from env_test import KukaPickPlaceEnv

env = KukaPickPlaceEnv(render_mode="human")  # or "rgb_array" or None
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Training

### Option 1: Pure PPO

Train a PPO agent from scratch:

```bash
python train_ppo.py --timesteps 500000 --n-envs 4

# With video recording and W&B logging
python train_ppo.py --timesteps 500000 --record-video --wandb
```

### Option 2: Behavioral Cloning + PPO (Recommended)

Use the IK teacher to bootstrap learning:

```bash
# Collect demonstrations, pretrain with BC, then fine-tune with PPO
python train_with_bc.py --collect-demos 100 --bc-epochs 50 --timesteps 200000
```

### Option 3: Step-by-Step Pipeline

```bash
# 1. Collect expert demonstrations
python collect_demos.py --episodes 100 --output demos/teacher_demos.npz

# 2. Pretrain policy with behavioral cloning
python bc_pretrain.py --demos demos/teacher_demos.npz --epochs 50

# 3. Fine-tune with PPO
python train_with_bc.py --demos demos/teacher_demos.npz --bc-epochs 0 --timesteps 200000
```

## Scripts Reference

| Script | Description |
|--------|-------------|
| `env_test.py` | Environment implementation and demo |
| `train_ppo.py` | Baseline PPO training script |
| `ik_teacher.py` | IK-based expert policy |
| `collect_demos.py` | Collect demonstrations using IK teacher |
| `bc_pretrain.py` | Behavioral cloning pretraining module |
| `train_with_bc.py` | Full BC + PPO training pipeline |
| `sanity_check.py` | Verify environment setup (useful for Docker) |

## Configuration

### PPO Hyperparameters

```bash
python train_ppo.py \
    --timesteps 500000 \
    --n-envs 4 \
    --lr 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    --n-epochs 5 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --ent-coef 0.01 \
    --normalize-reward
```

### Behavioral Cloning

```bash
python train_with_bc.py \
    --collect-demos 200 \
    --teacher-gain 1.0 \
    --bc-epochs 100 \
    --bc-batch-size 64 \
    --bc-lr 1e-3
```

## Logging & Monitoring

Training logs are saved to `./logs/` with the following structure:

```
logs/
└── ppo_kuka_20260121_143000/
    ├── tensorboard/      # TensorBoard logs
    ├── checkpoints/      # Model checkpoints
    ├── videos/           # Training videos (if enabled)
    ├── eval/             # Evaluation results and best model
    └── bc/               # BC pretraining artifacts (if used)
```

View training progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

## Docker Sanity Check

The sanity check verifies all components work correctly:

```bash
# Quick check (essential tests only)
docker run --rm kuka-mini-stacker python sanity_check.py --quick

# Full check (includes training test)
docker run --rm kuka-mini-stacker python sanity_check.py
```

Checks performed:
- Python version and dependencies
- URDF and mesh assets
- PyBullet headless connection
- Environment creation, reset, and step
- RGB rendering
- SB3 model creation and training
- Model save/load

## File Structure

```
kuka-mini-stacker/
├── env_test.py          # KukaPickPlaceEnv implementation
├── train_ppo.py         # PPO training script
├── ik_teacher.py        # IK teacher policy
├── collect_demos.py     # Demonstration collection
├── bc_pretrain.py       # Behavioral cloning module
├── train_with_bc.py     # BC + PPO pipeline
├── sanity_check.py      # Docker/environment validation
├── kuka_3dof.urdf       # Robot URDF definition
├── pybullet_kuka/       # KUKA mesh assets
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
└── logs/                # Training outputs
```

## Requirements

- Python >= 3.9
- gymnasium >= 1.2.0
- pybullet >= 3.2.5
- stable-baselines3 >= 2.3.0
- torch >= 2.0.0
- numpy >= 2.0.0

See `requirements.txt` for complete list.

## License

KUKA IIWA URDF and meshes are adapted from the RCPRG-ros-pkg/lwr_robot project, licensed under BSD-2-Clause.
