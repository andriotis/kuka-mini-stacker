"""
train_ppo.py - Train a PPO agent on the KUKA reaching task.

This script trains a baseline PPO agent using Stable Baselines3 with:
- CPU-first approach for pipeline validation
- Checkpointing every N steps
- Monitor wrapper for episode statistics
- TensorBoard logging (always enabled)
- W&B logging (optional, enabled via --wandb flag)
- Video recording during training
"""

import argparse
import os
from datetime import datetime
from typing import Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from env_test import KukaPickPlaceEnv


# Training constants
DEFAULT_TOTAL_TIMESTEPS = 50_000
DEFAULT_CHECKPOINT_FREQ = 10_000
DEFAULT_EVAL_FREQ = 5_000
DEFAULT_VIDEO_FREQ = 10_000
DEFAULT_VIDEO_LENGTH = 200
DEFAULT_N_EVAL_EPISODES = 5


def make_env(render_mode: str = None) -> Callable:
    """Create a factory function for the KUKA environment."""

    def _init() -> KukaPickPlaceEnv:
        return KukaPickPlaceEnv(render_mode=render_mode)

    return _init


def create_log_dir(base_dir: str = "./logs") -> dict:
    """Create timestamped log directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"ppo_kuka_{timestamp}")

    dirs = {
        "run": run_dir,
        "tensorboard": os.path.join(run_dir, "tensorboard"),
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "videos": os.path.join(run_dir, "videos"),
        "eval": os.path.join(run_dir, "eval"),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def setup_wandb(run_name: str, config: dict) -> "wandb.run":
    """Initialize Weights & Biases logging."""
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        run = wandb.init(
            project="kuka-mini-stacker",
            name=run_name,
            config=config,
            sync_tensorboard=True,
        )
        return run, WandbCallback
    except ImportError:
        print("Warning: wandb not installed. Skipping W&B logging.")
        return None, None


def train(args: argparse.Namespace) -> None:
    """Main training function."""
    print("=" * 60)
    print("PPO Training for KUKA Reaching Task")
    print("=" * 60)

    # Create log directories
    log_dirs = create_log_dir(args.log_dir)
    print(f"\nLog directory: {log_dirs['run']}")

    # Training configuration
    config = {
        "algorithm": "PPO",
        "policy": "MlpPolicy",
        "total_timesteps": args.timesteps,
        "learning_rate": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "clip_range_vf": args.clip_range_vf,
        "max_grad_norm": args.max_grad_norm,
        "normalize_reward": args.normalize_reward,
        "device": "cpu",
        "n_envs": args.n_envs,
    }

    # Initialize W&B if requested
    wandb_run = None
    wandb_callback = None
    if args.wandb:
        run_name = f"ppo_kuka_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_run, WandbCallbackClass = setup_wandb(run_name, config)
        if wandb_run and WandbCallbackClass:
            wandb_callback = WandbCallbackClass(
                model_save_path=os.path.join(log_dirs["checkpoints"], "wandb"),
                verbose=1,
            )

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create training environment with Monitor wrapper (auto-applied by make_vec_env)
    print("Creating training environment...")
    train_env = make_vec_env(
        make_env(render_mode="rgb_array"),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    # Optionally normalize observations and rewards for more stable training
    if args.normalize_reward:
        print("Applying reward normalization (VecNormalize)...")
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

    # Wrap with video recorder for training videos
    if args.record_video:
        train_env = VecVideoRecorder(
            train_env,
            video_folder=log_dirs["videos"],
            record_video_trigger=lambda step: step % args.video_freq == 0,
            video_length=args.video_length,
            name_prefix="training",
        )

    # Create separate evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        make_env(render_mode="rgb_array"),
        n_envs=1,
        seed=args.seed + 1000,
    )

    # Apply same normalization to eval env (using training stats)
    if args.normalize_reward:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,  # Don't normalize reward for evaluation
            clip_obs=10.0,
        )

    # Initialize PPO model
    print("Initializing PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        clip_range_vf=config["clip_range_vf"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        verbose=1,
        tensorboard_log=log_dirs["tensorboard"],
        device="cpu",
        seed=args.seed,
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=log_dirs["checkpoints"],
        name_prefix="ppo_kuka",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback - evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dirs["eval"],
        log_path=log_dirs["eval"],
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # Add W&B callback if enabled
    if wandb_callback:
        callbacks.append(wandb_callback)

    callback_list = CallbackList(callbacks)

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Checkpoint frequency: {args.checkpoint_freq} steps")
    print(f"Evaluation frequency: {args.eval_freq} steps")
    print("=" * 60 + "\n")

    # Check if progress bar dependencies are available
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401

        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
        print("Note: Install 'tqdm' and 'rich' for progress bar support.")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback_list,
            progress_bar=use_progress_bar,
            tb_log_name="PPO",
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save final model
        final_model_path = os.path.join(log_dirs["checkpoints"], "ppo_kuka_final")
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}.zip")

        # Cleanup
        train_env.close()
        eval_env.close()

        if wandb_run:
            import wandb

            wandb.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Logs: {log_dirs['run']}")
    print(f"TensorBoard: tensorboard --logdir {log_dirs['tensorboard']}")
    print(f"Best model: {log_dirs['eval']}/best_model.zip")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on KUKA reaching task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps per rollout",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Number of epochs per update",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda parameter",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range",
    )
    parser.add_argument(
        "--clip-range-vf",
        type=float,
        default=0.2,
        help="Value function clip range (None to disable)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for exploration",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function coefficient in loss",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--normalize-reward",
        action="store_true",
        help="Normalize observations and rewards with VecNormalize",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Base directory for logs",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=DEFAULT_CHECKPOINT_FREQ,
        help="Checkpoint save frequency (steps)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=DEFAULT_EVAL_FREQ,
        help="Evaluation frequency (steps)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=DEFAULT_N_EVAL_EPISODES,
        help="Number of episodes per evaluation",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    # Video recording
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record training videos",
    )
    parser.add_argument(
        "--video-freq",
        type=int,
        default=DEFAULT_VIDEO_FREQ,
        help="Video recording frequency (steps)",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=DEFAULT_VIDEO_LENGTH,
        help="Video length (steps)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
