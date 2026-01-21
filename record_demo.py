"""
record_demo.py - Record a demo video of a trained PPO agent.

This script loads a trained checkpoint and records the agent's behavior
as an MP4 video or GIF for demonstration purposes.
"""

import argparse
import os

import imageio
import numpy as np
from stable_baselines3 import PPO

from env_test import KukaPickPlaceEnv


# Recording constants
DEFAULT_NUM_EPISODES = 3
DEFAULT_MAX_STEPS_PER_EPISODE = 200
DEFAULT_FPS = 30
DEFAULT_OUTPUT_PATH = "./demo.mp4"


def record_demo(
    model_path: str,
    output_path: str,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS_PER_EPISODE,
    fps: int = DEFAULT_FPS,
    deterministic: bool = True,
) -> None:
    """
    Record a demo video of the trained agent.

    Args:
        model_path: Path to the trained model (.zip file)
        output_path: Output path for the video (supports .mp4 and .gif)
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        fps: Frames per second for the output video
        deterministic: Use deterministic actions (no exploration)
    """
    print("=" * 60)
    print("Recording Demo Video")
    print("=" * 60)

    # Load the trained model
    print(f"\nLoading model from: {model_path}")
    if not os.path.exists(model_path) and not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = PPO.load(model_path)
    print("Model loaded successfully!")

    # Create environment with rgb_array rendering
    print("Creating environment...")
    env = KukaPickPlaceEnv(render_mode="rgb_array")

    # Collect frames
    frames = []
    total_rewards = []
    successes = 0

    print(f"\nRecording {num_episodes} episodes...")
    print("-" * 40)

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_frames = []

        for step in range(max_steps):
            # Capture frame
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)

            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                # Capture final frame
                final_frame = env.render()
                if final_frame is not None:
                    episode_frames.append(final_frame)
                break

        # Track results
        total_rewards.append(episode_reward)
        if terminated and info.get("distance_to_target", float("inf")) < 0.05:
            successes += 1

        frames.extend(episode_frames)

        # Add a brief pause between episodes (black frames)
        if episode < num_episodes - 1 and episode_frames:
            pause_frame = np.zeros_like(episode_frames[0])
            frames.extend([pause_frame] * int(fps * 0.5))  # 0.5 second pause

        print(
            f"Episode {episode + 1}/{num_episodes}: "
            f"Reward = {episode_reward:.2f}, "
            f"Steps = {step + 1}, "
            f"Distance = {info.get('distance_to_target', 'N/A'):.3f}"
        )

    env.close()

    # Print summary
    print("-" * 40)
    print(f"\nSummary:")
    print(f"  Mean reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"  Total frames: {len(frames)}")

    # Save video
    if not frames:
        print("\nError: No frames captured!")
        return

    print(f"\nSaving video to: {output_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Determine output format and save
    if output_path.lower().endswith(".gif"):
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
    else:
        # Default to MP4
        imageio.mimsave(output_path, frames, fps=fps, codec="libx264", pixelformat="yuv420p")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Video saved! (Size: {file_size_mb:.2f} MB)")

    print("\n" + "=" * 60)
    print("Demo recording complete!")
    print("=" * 60)


def find_best_model(log_dir: str = "./logs") -> str:
    """Find the best model from the most recent training run."""
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Find most recent run directory
    run_dirs = [
        d for d in os.listdir(log_dir)
        if d.startswith("ppo_kuka_") and os.path.isdir(os.path.join(log_dir, d))
    ]

    if not run_dirs:
        raise FileNotFoundError(f"No training runs found in {log_dir}")

    # Sort by name (timestamp) and get most recent
    run_dirs.sort(reverse=True)
    latest_run = run_dirs[0]

    # Look for best model first, then final model
    eval_dir = os.path.join(log_dir, latest_run, "eval")
    best_model = os.path.join(eval_dir, "best_model.zip")
    if os.path.exists(best_model):
        return best_model

    # Fall back to final checkpoint
    checkpoints_dir = os.path.join(log_dir, latest_run, "checkpoints")
    final_model = os.path.join(checkpoints_dir, "ppo_kuka_final.zip")
    if os.path.exists(final_model):
        return final_model

    # Look for any checkpoint
    if os.path.exists(checkpoints_dir):
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".zip")]
        if checkpoints:
            checkpoints.sort(reverse=True)
            return os.path.join(checkpoints_dir, checkpoints[0])

    raise FileNotFoundError(f"No model found in {os.path.join(log_dir, latest_run)}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Record demo video of trained PPO agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model. If not specified, uses latest best model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output video path (.mp4 or .gif)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_EPISODE,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Output video FPS",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic (non-deterministic) actions",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Log directory to search for models",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Find model path
    if args.model:
        model_path = args.model
    else:
        print("No model specified, searching for latest best model...")
        try:
            model_path = find_best_model(args.log_dir)
            print(f"Found model: {model_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("\nPlease specify a model path with --model or run training first.")
            exit(1)

    record_demo(
        model_path=model_path,
        output_path=args.output,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        deterministic=not args.stochastic,
    )
