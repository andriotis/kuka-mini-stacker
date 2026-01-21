"""
train_with_bc.py - Train PPO with behavioral cloning pretraining.

This script implements the full training pipeline:
1. Collect demonstrations using IK teacher (optional)
2. Pretrain PPO policy with behavioral cloning
3. Continue training with standard PPO

The BC pretraining helps bootstrap learning by initializing the policy
to imitate expert behavior before reinforcement learning refinement.
"""

import argparse
import os
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder

from env_test import KukaPickPlaceEnv
from collect_demos import collect_demonstrations, load_demonstrations
from bc_pretrain import bc_pretrain, evaluate_bc_policy


# Training constants (same as train_ppo.py)
DEFAULT_TOTAL_TIMESTEPS = 50_000
DEFAULT_CHECKPOINT_FREQ = 10_000
DEFAULT_EVAL_FREQ = 5_000
DEFAULT_VIDEO_FREQ = 10_000
DEFAULT_VIDEO_LENGTH = 200
DEFAULT_N_EVAL_EPISODES = 5

# BC-specific constants
DEFAULT_BC_EPOCHS = 50
DEFAULT_BC_BATCH_SIZE = 64
DEFAULT_BC_LEARNING_RATE = 1e-3
DEFAULT_DEMO_EPISODES = 100
DEFAULT_DEMO_PATH = "./demos/teacher_demos.npz"


def make_env(render_mode: str = None) -> Callable:
    """Create a factory function for the KUKA environment."""

    def _init() -> KukaPickPlaceEnv:
        return KukaPickPlaceEnv(render_mode=render_mode)

    return _init


def create_log_dir(base_dir: str = "./logs") -> dict:
    """Create timestamped log directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"ppo_bc_kuka_{timestamp}")

    dirs = {
        "run": run_dir,
        "tensorboard": os.path.join(run_dir, "tensorboard"),
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "videos": os.path.join(run_dir, "videos"),
        "eval": os.path.join(run_dir, "eval"),
        "bc": os.path.join(run_dir, "bc"),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def setup_wandb(run_name: str, config: dict):
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


def train_with_bc(args: argparse.Namespace) -> None:
    """Main training function with BC pretraining."""
    print("=" * 60)
    print("PPO Training with Behavioral Cloning Pretraining")
    print("=" * 60)

    # Create log directories
    log_dirs = create_log_dir(args.log_dir)
    print(f"\nLog directory: {log_dirs['run']}")

    # =========================================================================
    # Phase 1: Demonstration Collection (if needed)
    # =========================================================================
    demo_path = args.demos

    if args.collect_demos > 0:
        print("\n" + "=" * 60)
        print("Phase 1: Collecting Demonstrations")
        print("=" * 60)

        demo_path = os.path.join(log_dirs["bc"], "teacher_demos.npz")
        collect_demonstrations(
            num_episodes=args.collect_demos,
            output_path=demo_path,
            seed=args.seed,
            teacher_gain=args.teacher_gain,
            verbose=True,
        )
    elif demo_path and os.path.exists(demo_path):
        print(f"\nUsing existing demonstrations: {demo_path}")
        demo_data = load_demonstrations(demo_path)
        if demo_data.get("metadata"):
            print(f"  Transitions: {len(demo_data['observations'])}")
            meta = demo_data["metadata"]
            if isinstance(meta, dict):
                print(f"  Success rate: {meta.get('success_rate', 'N/A')}")
    else:
        print(f"\nNo demonstrations found at {demo_path}")
        print("Run with --collect-demos N to collect demonstrations first.")
        return

    # =========================================================================
    # Phase 2: Create Environment and Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Creating Environment and Model")
    print("=" * 60)

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
        # BC-specific config
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "bc_learning_rate": args.bc_lr,
    }

    # Initialize W&B if requested
    wandb_run = None
    wandb_callback = None
    if args.wandb:
        run_name = f"ppo_bc_kuka_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    # Create training environment
    print("Creating training environment...")
    train_env = make_vec_env(
        make_env(render_mode="rgb_array"),
        n_envs=args.n_envs,
        seed=args.seed,
    )

    if args.normalize_reward:
        print("Applying reward normalization (VecNormalize)...")
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

    if args.record_video:
        train_env = VecVideoRecorder(
            train_env,
            video_folder=log_dirs["videos"],
            record_video_trigger=lambda step: step % args.video_freq == 0,
            video_length=args.video_length,
            name_prefix="training",
        )

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        make_env(render_mode="rgb_array"),
        n_envs=1,
        seed=args.seed + 1000,
    )

    if args.normalize_reward:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
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

    # =========================================================================
    # Phase 3: Behavioral Cloning Pretraining
    # =========================================================================
    if args.bc_epochs > 0:
        print("\n" + "=" * 60)
        print("Phase 3: Behavioral Cloning Pretraining")
        print("=" * 60)

        # Load demonstrations
        demo_data = load_demonstrations(demo_path)

        # Run BC pretraining
        bc_history = bc_pretrain(
            model,
            demo_data["observations"],
            demo_data["actions"],
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            learning_rate=args.bc_lr,
            verbose=True,
        )

        # Save BC training history
        bc_history_path = os.path.join(log_dirs["bc"], "bc_history.npz")
        np.savez(bc_history_path, **bc_history)
        print(f"BC history saved to: {bc_history_path}")

        # Save BC-pretrained model
        bc_model_path = os.path.join(log_dirs["bc"], "bc_pretrained_model")
        model.save(bc_model_path)
        print(f"BC-pretrained model saved to: {bc_model_path}.zip")

        # Evaluate BC-pretrained policy
        print("\nEvaluating BC-pretrained policy...")
        bc_results = evaluate_bc_policy(
            model, num_episodes=args.n_eval_episodes, seed=args.seed
        )

        # Log BC results to W&B if enabled
        if wandb_run:
            import wandb
            wandb.log({
                "bc/final_train_loss": bc_history["train_loss"][-1],
                "bc/final_val_loss": bc_history["val_loss"][-1],
                "bc/eval_mean_reward": bc_results["mean_reward"],
                "bc/eval_success_rate": bc_results["success_rate"],
            })
    else:
        print("\nSkipping BC pretraining (--bc-epochs 0)")

    # =========================================================================
    # Phase 4: PPO Training
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 4: PPO Reinforcement Learning")
    print("=" * 60)

    # Setup callbacks
    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=log_dirs["checkpoints"],
        name_prefix="ppo_bc_kuka",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

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

    if wandb_callback:
        callbacks.append(wandb_callback)

    callback_list = CallbackList(callbacks)

    # Start training
    print(f"\nTotal timesteps: {args.timesteps}")
    print(f"Checkpoint frequency: {args.checkpoint_freq} steps")
    print(f"Evaluation frequency: {args.eval_freq} steps")
    print("=" * 60 + "\n")

    # Check for progress bar support
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
            tb_log_name="PPO_BC",
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save final model
        final_model_path = os.path.join(log_dirs["checkpoints"], "ppo_bc_kuka_final")
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
        description="Train PPO with behavioral cloning pretraining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Demo collection
    parser.add_argument(
        "--demos",
        type=str,
        default=DEFAULT_DEMO_PATH,
        help="Path to demonstration file (.npz)",
    )
    parser.add_argument(
        "--collect-demos",
        type=int,
        default=0,
        help="Number of episodes to collect (0 = use existing)",
    )
    parser.add_argument(
        "--teacher-gain",
        type=float,
        default=1.0,
        help="IK teacher gain for demo collection",
    )

    # BC pretraining
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=DEFAULT_BC_EPOCHS,
        help="Number of BC pretraining epochs (0 to skip)",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=DEFAULT_BC_BATCH_SIZE,
        help="Batch size for BC pretraining",
    )
    parser.add_argument(
        "--bc-lr",
        type=float,
        default=DEFAULT_BC_LEARNING_RATE,
        help="Learning rate for BC pretraining",
    )

    # PPO training parameters (same as train_ppo.py)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Total PPO training timesteps",
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
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch size")
    parser.add_argument("--n-epochs", type=int, default=5, help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--clip-range-vf", type=float, default=0.2, help="VF clip range")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--normalize-reward", action="store_true", help="Use VecNormalize")

    # Logging
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--n-eval-episodes", type=int, default=DEFAULT_N_EVAL_EPISODES)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # Video
    parser.add_argument("--record-video", action="store_true", help="Record videos")
    parser.add_argument("--video-freq", type=int, default=DEFAULT_VIDEO_FREQ)
    parser.add_argument("--video-length", type=int, default=DEFAULT_VIDEO_LENGTH)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_with_bc(args)
