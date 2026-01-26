"""
compare_ik.py - Launch joint-space and IK training in parallel for comparison.

This script starts two training runs simultaneously:
1. Joint-space control (baseline)
2. IK-based Cartesian control

Both runs log to the same base directory, allowing side-by-side comparison
in TensorBoard.

Usage:
    python compare_ik.py --timesteps 50000

    # In another terminal:
    tensorboard --logdir ./logs/comparison
"""

import argparse
import subprocess
import sys
from datetime import datetime


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare joint-space vs IK control for KUKA reaching task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total training timesteps for each run",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/comparison",
        help="Base directory for logs (both runs will log here)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments per training run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (IK run uses seed+1 for variety)",
    )

    return parser.parse_args()


def main():
    """Launch both training runs in parallel."""
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("IK vs Joint-Space Comparison Training")
    print("=" * 60)
    print(f"\nTimesteps per run: {args.timesteps:,}")
    print(f"Log directory: {args.log_dir}")
    print(f"Parallel envs per run: {args.n_envs}")
    print(f"\nStarting both training runs...")
    print("=" * 60)

    # Common arguments for both runs
    common_args = [
        "--timesteps", str(args.timesteps),
        "--log-dir", args.log_dir,
        "--n-envs", str(args.n_envs),
    ]

    # Joint-space training command
    joint_cmd = [
        sys.executable, "train_ppo.py",
        *common_args,
        "--seed", str(args.seed),
    ]

    # IK training command
    ik_cmd = [
        sys.executable, "train_ppo.py",
        *common_args,
        "--use-ik",
        "--seed", str(args.seed + 1),
    ]

    print(f"\n[Joint-space] Starting: {' '.join(joint_cmd)}")
    print(f"[IK mode] Starting: {' '.join(ik_cmd)}")
    print()

    # Launch both processes
    joint_process = subprocess.Popen(joint_cmd)
    ik_process = subprocess.Popen(ik_cmd)

    print("Both training runs started!")
    print(f"\nTo view live comparison, run in another terminal:")
    print(f"  tensorboard --logdir {args.log_dir}")
    print()

    # Wait for both to complete
    try:
        joint_exit = joint_process.wait()
        ik_exit = ik_process.wait()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Joint-space exit code: {joint_exit}")
        print(f"IK mode exit code: {ik_exit}")
        print(f"\nView results: tensorboard --logdir {args.log_dir}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Terminating both processes...")
        joint_process.terminate()
        ik_process.terminate()
        joint_process.wait()
        ik_process.wait()
        print("Both processes terminated.")


if __name__ == "__main__":
    main()
