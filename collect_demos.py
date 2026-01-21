"""
collect_demos.py - Collect demonstration trajectories using the IK teacher.

This script uses the IK teacher policy to collect expert demonstrations
for behavioral cloning pretraining.
"""

import argparse
import os
from datetime import datetime

import numpy as np

from env_test import KukaPickPlaceEnv
from ik_teacher import IKTeacherWithEnv


# Default configuration
DEFAULT_NUM_EPISODES = 100
DEFAULT_MAX_STEPS = 200
DEFAULT_OUTPUT_DIR = "./demos"
DEFAULT_OUTPUT_FILE = "teacher_demos.npz"


def collect_demonstrations(
    num_episodes: int = DEFAULT_NUM_EPISODES,
    max_steps_per_episode: int = DEFAULT_MAX_STEPS,
    output_path: str = None,
    seed: int = 42,
    teacher_gain: float = 1.0,
    verbose: bool = True,
) -> dict:
    """
    Collect demonstration trajectories using the IK teacher.

    Args:
        num_episodes: Number of episodes to collect.
        max_steps_per_episode: Maximum steps per episode.
        output_path: Path to save the demonstrations (.npz file).
        seed: Random seed for reproducibility.
        teacher_gain: Gain factor for IK teacher (higher = more aggressive).
        verbose: Whether to print progress.

    Returns:
        Dictionary containing the collected demonstrations.
    """
    if verbose:
        print("=" * 60)
        print("Collecting IK Teacher Demonstrations")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Max steps per episode: {max_steps_per_episode}")
        print(f"  Teacher gain: {teacher_gain}")
        print(f"  Seed: {seed}")
        print()

    # Create environment and teacher
    env = KukaPickPlaceEnv(render_mode=None, max_episode_steps=max_steps_per_episode)
    teacher = IKTeacherWithEnv(env, gain=teacher_gain)

    # Storage lists
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    # Statistics
    episode_rewards = []
    episode_lengths = []
    successes = 0

    try:
        for episode in range(num_episodes):
            obs, info = env.reset(seed=seed + episode)
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                # Get teacher action
                action = teacher.get_action(obs)

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store transition
                observations.append(obs.copy())
                actions.append(action.copy())
                rewards.append(reward)
                next_observations.append(next_obs.copy())
                dones.append(done)

                # Update for next iteration
                obs = next_obs
                episode_reward += reward
                episode_length += 1

            # Track episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if info.get("distance_to_target", float("inf")) < 0.05:
                successes += 1

            if verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
                print(
                    f"Episode {episode + 1}/{num_episodes}: "
                    f"Reward = {episode_reward:.2f}, "
                    f"Steps = {episode_length}, "
                    f"Distance = {info.get('distance_to_target', 'N/A'):.4f}"
                )

    finally:
        env.close()

    # Convert to numpy arrays
    demo_data = {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_observations": np.array(next_observations, dtype=np.float32),
        "dones": np.array(dones, dtype=bool),
    }

    # Add metadata
    demo_data["metadata"] = np.array([{
        "num_episodes": num_episodes,
        "total_transitions": len(observations),
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "success_rate": successes / num_episodes,
        "teacher_gain": teacher_gain,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }])

    if verbose:
        print("\n" + "-" * 60)
        print("Collection Summary:")
        print(f"  Total transitions: {len(observations)}")
        print(f"  Mean episode reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
        print(f"  Mean episode length: {np.mean(episode_lengths):.1f}")
        print(f"  Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")

    # Save to file if path provided
    if output_path:
        # Create directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.savez_compressed(output_path, **demo_data)

        if verbose:
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\nDemonstrations saved to: {output_path}")
            print(f"File size: {file_size_mb:.2f} MB")

    if verbose:
        print("=" * 60)

    return demo_data


def load_demonstrations(path: str) -> dict:
    """
    Load demonstrations from a .npz file.

    Args:
        path: Path to the .npz file.

    Returns:
        Dictionary containing the demonstrations.
    """
    data = np.load(path, allow_pickle=True)
    return {
        "observations": data["observations"],
        "actions": data["actions"],
        "rewards": data["rewards"],
        "next_observations": data["next_observations"],
        "dones": data["dones"],
        "metadata": data["metadata"].item() if "metadata" in data else {},
    }


def print_demo_info(path: str) -> None:
    """Print information about a demonstration file."""
    data = load_demonstrations(path)

    print(f"\nDemonstration file: {path}")
    print("-" * 40)
    print(f"Observations shape: {data['observations'].shape}")
    print(f"Actions shape: {data['actions'].shape}")
    print(f"Rewards shape: {data['rewards'].shape}")
    print(f"Dones shape: {data['dones'].shape}")

    if data["metadata"]:
        print("\nMetadata:")
        for key, value in data["metadata"].items():
            print(f"  {key}: {value}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect IK teacher demonstrations for behavioral cloning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_OUTPUT_FILE),
        help="Output path for the .npz file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--teacher-gain",
        type=float,
        default=1.0,
        help="IK teacher gain (higher = more aggressive movements)",
    )
    parser.add_argument(
        "--info",
        type=str,
        default=None,
        help="Print info about an existing demo file instead of collecting",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.info:
        # Print info about existing file
        print_demo_info(args.info)
    else:
        # Collect new demonstrations
        collect_demonstrations(
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            output_path=args.output,
            seed=args.seed,
            teacher_gain=args.teacher_gain,
            verbose=not args.quiet,
        )
