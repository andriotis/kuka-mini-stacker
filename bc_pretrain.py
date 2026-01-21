"""
bc_pretrain.py - Behavioral cloning pretraining for PPO policy.

This module provides functionality to pretrain a PPO policy network
using supervised learning on expert demonstrations (behavioral cloning).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Callable

from stable_baselines3 import PPO


# Default BC hyperparameters
DEFAULT_BC_EPOCHS = 50
DEFAULT_BC_BATCH_SIZE = 64
DEFAULT_BC_LEARNING_RATE = 1e-3
DEFAULT_VALIDATION_SPLIT = 0.1


class DemonstrationDataset:
    """
    Dataset wrapper for demonstration data.

    Handles loading, preprocessing, and splitting of demonstration data
    for behavioral cloning training.
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        validation_split: float = DEFAULT_VALIDATION_SPLIT,
        device: str = "cpu",
    ):
        """
        Initialize the demonstration dataset.

        Args:
            observations: Array of observations, shape (N, obs_dim).
            actions: Array of actions, shape (N, action_dim).
            validation_split: Fraction of data to use for validation.
            device: Torch device to use.
        """
        self.device = device

        # Convert to tensors
        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
        action_tensor = torch.tensor(actions, dtype=torch.float32, device=device)

        # Split into train and validation
        n_samples = len(observations)
        n_val = int(n_samples * validation_split)
        n_train = n_samples - n_val

        # Shuffle indices
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Create datasets
        self.train_obs = obs_tensor[train_indices]
        self.train_actions = action_tensor[train_indices]
        self.val_obs = obs_tensor[val_indices]
        self.val_actions = action_tensor[val_indices]

        self.n_train = n_train
        self.n_val = n_val

    def get_train_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """Get a DataLoader for training data."""
        dataset = TensorDataset(self.train_obs, self.train_actions)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_val_dataloader(self, batch_size: int) -> DataLoader:
        """Get a DataLoader for validation data."""
        dataset = TensorDataset(self.val_obs, self.val_actions)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_policy_action(model: PPO, observations: torch.Tensor) -> torch.Tensor:
    """
    Get the deterministic action from the PPO policy.

    This function extracts the mean action (deterministic) from the policy
    network without sampling from the distribution.

    Args:
        model: The PPO model.
        observations: Batch of observations.

    Returns:
        Batch of predicted actions.
    """
    # Get the policy's action distribution parameters
    # The MlpPolicy uses a Gaussian distribution with learned mean and std
    features = model.policy.extract_features(observations)

    if model.policy.share_features_extractor:
        latent_pi, _ = model.policy.mlp_extractor(features)
    else:
        latent_pi = model.policy.mlp_extractor.forward_actor(features)

    # Get the mean action (deterministic)
    mean_actions = model.policy.action_net(latent_pi)

    return mean_actions


def bc_pretrain(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int = DEFAULT_BC_EPOCHS,
    batch_size: int = DEFAULT_BC_BATCH_SIZE,
    learning_rate: float = DEFAULT_BC_LEARNING_RATE,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    verbose: bool = True,
    callback: Optional[Callable[[int, float, float], None]] = None,
) -> dict:
    """
    Pretrain a PPO policy using behavioral cloning.

    This function trains only the actor network (policy) using MSE loss
    to imitate expert demonstrations. The value function is not trained
    and will learn from PPO.

    Args:
        model: The PPO model to pretrain.
        observations: Expert observations, shape (N, obs_dim).
        actions: Expert actions, shape (N, action_dim).
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        validation_split: Fraction of data for validation.
        verbose: Whether to print progress.
        callback: Optional callback function called after each epoch
                  with (epoch, train_loss, val_loss).

    Returns:
        Dictionary with training history.
    """
    if verbose:
        print("=" * 60)
        print("Behavioral Cloning Pretraining")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Validation split: {validation_split}")
        print(f"  Demonstrations: {len(observations)}")
        print()

    device = model.device

    # Create dataset
    dataset = DemonstrationDataset(
        observations, actions, validation_split=validation_split, device=device
    )

    if verbose:
        print(f"Training samples: {dataset.n_train}")
        print(f"Validation samples: {dataset.n_val}")
        print()

    # Create dataloaders
    train_loader = dataset.get_train_dataloader(batch_size)
    val_loader = dataset.get_val_dataloader(batch_size)

    # Collect actor parameters to optimize
    # We only train the actor (policy) network, not the value function
    actor_params = list(model.policy.mlp_extractor.policy_net.parameters())
    actor_params += list(model.policy.action_net.parameters())

    # Also include the shared feature extractor if it exists
    if hasattr(model.policy, "features_extractor"):
        actor_params += list(model.policy.features_extractor.parameters())

    # Create optimizer
    optimizer = torch.optim.Adam(actor_params, lr=learning_rate)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    # Training loop
    model.policy.train()

    for epoch in range(epochs):
        # Training phase
        train_losses = []

        for obs_batch, action_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            pred_actions = get_policy_action(model, obs_batch)

            # Compute MSE loss
            loss = F.mse_loss(pred_actions, action_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.policy.eval()
        val_losses = []

        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                pred_actions = get_policy_action(model, obs_batch)
                loss = F.mse_loss(pred_actions, action_batch)
                val_losses.append(loss.item())

        model.policy.train()

        # Record history
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Callback
        if callback:
            callback(epoch, train_loss, val_loss)

        # Print progress
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss = {train_loss:.6f}, "
                f"Val Loss = {val_loss:.6f}"
            )

    # Set policy back to eval mode
    model.policy.eval()

    if verbose:
        print("\n" + "-" * 60)
        print("BC Pretraining Complete!")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")
        print("=" * 60)

    return history


def bc_pretrain_from_file(
    model: PPO,
    demo_path: str,
    **kwargs,
) -> dict:
    """
    Pretrain a PPO policy from a demonstration file.

    Args:
        model: The PPO model to pretrain.
        demo_path: Path to the .npz demonstration file.
        **kwargs: Additional arguments passed to bc_pretrain.

    Returns:
        Dictionary with training history.
    """
    # Load demonstrations
    data = np.load(demo_path, allow_pickle=True)
    observations = data["observations"]
    actions = data["actions"]

    return bc_pretrain(model, observations, actions, **kwargs)


def evaluate_bc_policy(
    model: PPO,
    num_episodes: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the BC-pretrained policy on the environment.

    Args:
        model: The PPO model to evaluate.
        num_episodes: Number of episodes to run.
        seed: Random seed.
        verbose: Whether to print results.

    Returns:
        Dictionary with evaluation metrics.
    """
    from env_test import KukaPickPlaceEnv

    env = KukaPickPlaceEnv(render_mode=None)

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
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if info.get("distance_to_target", float("inf")) < 0.05:
                successes += 1

    finally:
        env.close()

    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": successes / num_episodes,
    }

    if verbose:
        print("\nBC Policy Evaluation:")
        print(f"  Mean reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"  Mean length: {results['mean_length']:.1f}")
        print(f"  Success rate: {successes}/{num_episodes} ({100*results['success_rate']:.1f}%)")

    return results


if __name__ == "__main__":
    import argparse
    from stable_baselines3.common.env_util import make_vec_env
    from env_test import KukaPickPlaceEnv

    parser = argparse.ArgumentParser(description="BC pretraining standalone test")
    parser.add_argument("--demos", type=str, required=True, help="Path to demo file")
    parser.add_argument("--epochs", type=int, default=DEFAULT_BC_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BC_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_BC_LEARNING_RATE)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    # Create environment and model
    print("Creating environment and PPO model...")
    env = make_vec_env(lambda: KukaPickPlaceEnv(render_mode=None), n_envs=1)
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")

    # Run BC pretraining
    history = bc_pretrain_from_file(
        model,
        args.demos,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Evaluate
    evaluate_bc_policy(model, num_episodes=args.eval_episodes)

    env.close()
