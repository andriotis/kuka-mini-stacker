"""
ik_teacher.py - IK-based teacher policy for the KUKA reaching task.

This module provides an IK teacher that uses PyBullet's calculateInverseKinematics
to generate expert demonstrations for behavioral cloning.
"""

import numpy as np
import pybullet as p

from env_test import (
    KUKA_ARM_JOINT_COUNT,
    KUKA_EE_LINK_INDEX,
    KUKA_JOINT_DELTA_SCALE,
    KUKA_ACTION_DIM,
)


# Observation indices for the 27-D observation vector
OBS_SIN_Q_START = 0
OBS_SIN_Q_END = 4
OBS_COS_Q_START = 4
OBS_COS_Q_END = 8
OBS_QDOT_START = 8
OBS_QDOT_END = 12
OBS_GRIP_POS_START = 12
OBS_GRIP_POS_END = 14
OBS_GRIP_VEL_START = 14
OBS_GRIP_VEL_END = 16
OBS_RELATIVE_TARGET_START = 16
OBS_RELATIVE_TARGET_END = 19
OBS_EE_POS_START = 19
OBS_EE_POS_END = 22
OBS_PREV_ACTION_START = 22
OBS_PREV_ACTION_END = 27


class IKTeacherPolicy:
    """
    IK-based teacher policy that computes actions to move the end-effector
    toward the target position using inverse kinematics.

    This policy requires access to the PyBullet physics client and robot ID
    from the environment to compute IK solutions.
    """

    def __init__(
        self,
        gain: float = 1.0,
        use_nullspace: bool = True,
        max_iterations: int = 100,
        residual_threshold: float = 1e-4,
    ):
        """
        Initialize the IK teacher policy.

        Args:
            gain: Scaling factor for joint deltas (higher = more aggressive).
            use_nullspace: Whether to use nullspace IK for better solutions.
            max_iterations: Maximum IK solver iterations.
            residual_threshold: IK convergence threshold.
        """
        self.gain = gain
        self.use_nullspace = use_nullspace
        self.max_iterations = max_iterations
        self.residual_threshold = residual_threshold

        # Joint limits for nullspace IK (from URDF)
        self.joint_lower_limits = [-2.967, -2.094, -2.967, -2.094]
        self.joint_upper_limits = [2.967, 2.094, 2.967, 2.094]
        self.joint_ranges = [5.934, 4.188, 5.934, 4.188]
        self.rest_poses = [0.0, 0.0, 0.0, 0.0]

    def extract_joint_angles_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract current joint angles from the observation using sin/cos encoding.

        Args:
            obs: The 27-D observation vector.

        Returns:
            Array of 4 joint angles in radians.
        """
        sin_q = obs[OBS_SIN_Q_START:OBS_SIN_Q_END]
        cos_q = obs[OBS_COS_Q_START:OBS_COS_Q_END]
        joint_angles = np.arctan2(sin_q, cos_q)
        return joint_angles

    def extract_target_position(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract the target position from the observation.

        The target position is computed as: ee_pos + relative_target

        Args:
            obs: The 27-D observation vector.

        Returns:
            3D target position in world coordinates.
        """
        relative_target = obs[OBS_RELATIVE_TARGET_START:OBS_RELATIVE_TARGET_END]
        ee_pos = obs[OBS_EE_POS_START:OBS_EE_POS_END]
        target_pos = ee_pos + relative_target
        return target_pos

    def extract_ee_position(self, obs: np.ndarray) -> np.ndarray:
        """
        Extract the end-effector position from the observation.

        Args:
            obs: The 27-D observation vector.

        Returns:
            3D end-effector position in world coordinates.
        """
        return obs[OBS_EE_POS_START:OBS_EE_POS_END].copy()

    def compute_ik_action(
        self,
        obs: np.ndarray,
        robot_id: int,
        arm_joint_indices: list,
    ) -> np.ndarray:
        """
        Compute the IK-based action to move toward the target.

        Args:
            obs: The 27-D observation vector.
            robot_id: PyBullet body ID of the robot.
            arm_joint_indices: List of joint indices for the arm.

        Returns:
            5-D action vector (4 arm joint deltas + 1 gripper action).
        """
        # Extract current state from observation
        current_joints = self.extract_joint_angles_from_obs(obs)
        target_pos = self.extract_target_position(obs)

        # Compute IK solution
        if self.use_nullspace:
            # Use nullspace IK for more stable solutions
            # Note: We need to provide limits for ALL joints the IK might consider
            # PyBullet IK returns values for joints up to the end-effector
            ik_solution = p.calculateInverseKinematics(
                robot_id,
                KUKA_EE_LINK_INDEX,
                target_pos,
                lowerLimits=self.joint_lower_limits,
                upperLimits=self.joint_upper_limits,
                jointRanges=self.joint_ranges,
                restPoses=self.rest_poses,
                maxNumIterations=self.max_iterations,
                residualThreshold=self.residual_threshold,
            )
        else:
            # Simple IK without nullspace constraints
            ik_solution = p.calculateInverseKinematics(
                robot_id,
                KUKA_EE_LINK_INDEX,
                target_pos,
                maxNumIterations=self.max_iterations,
                residualThreshold=self.residual_threshold,
            )

        # Extract joint targets for the 4 controlled arm joints
        # IK returns values for all joints, we only need the first 4
        target_joints = np.array(ik_solution[:KUKA_ARM_JOINT_COUNT], dtype=np.float32)

        # Compute joint deltas
        joint_deltas = target_joints - current_joints

        # Scale deltas to action space [-1, 1]
        # The environment scales actions by KUKA_JOINT_DELTA_SCALE
        scaled_deltas = (joint_deltas / KUKA_JOINT_DELTA_SCALE) * self.gain

        # Clip to valid action range
        arm_actions = np.clip(scaled_deltas, -1.0, 1.0)

        # Create full action vector (4 arm + 1 gripper)
        # Gripper action is 0 (neutral) for reaching task
        action = np.zeros(KUKA_ACTION_DIM, dtype=np.float32)
        action[:KUKA_ARM_JOINT_COUNT] = arm_actions
        action[4] = 0.0  # Gripper neutral

        return action

    def predict(
        self,
        obs: np.ndarray,
        robot_id: int,
        arm_joint_indices: list,
        deterministic: bool = True,
    ) -> tuple:
        """
        Predict action given observation (compatible with SB3 policy interface).

        Args:
            obs: The observation (can be batched or single).
            robot_id: PyBullet body ID of the robot.
            arm_joint_indices: List of joint indices for the arm.
            deterministic: Ignored (teacher is always deterministic).

        Returns:
            Tuple of (action, None) to match SB3 interface.
        """
        # Handle batched observations
        if obs.ndim == 2:
            actions = np.array([
                self.compute_ik_action(o, robot_id, arm_joint_indices)
                for o in obs
            ])
            return actions, None

        # Single observation
        action = self.compute_ik_action(obs, robot_id, arm_joint_indices)
        return action, None


class IKTeacherWithEnv:
    """
    Convenience wrapper that holds both the IK teacher and environment reference.

    This class makes it easier to use the IK teacher without passing
    robot_id and joint indices every time.
    """

    def __init__(self, env, gain: float = 1.0, **kwargs):
        """
        Initialize the IK teacher with an environment reference.

        Args:
            env: The KukaPickPlaceEnv instance.
            gain: Scaling factor for joint deltas.
            **kwargs: Additional arguments passed to IKTeacherPolicy.
        """
        self.env = env
        self.teacher = IKTeacherPolicy(gain=gain, **kwargs)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple:
        """
        Predict action given observation.

        Args:
            obs: The observation.
            deterministic: Ignored (teacher is always deterministic).

        Returns:
            Tuple of (action, None) to match SB3 interface.
        """
        return self.teacher.predict(
            obs,
            self.env.robot_id,
            self.env.arm_joint_indices,
            deterministic=deterministic,
        )

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get action for a single observation.

        Args:
            obs: The observation.

        Returns:
            The action array.
        """
        action, _ = self.predict(obs)
        return action


def test_ik_teacher():
    """Test the IK teacher policy with the KUKA environment."""
    from env_test import KukaPickPlaceEnv

    print("=" * 60)
    print("Testing IK Teacher Policy")
    print("=" * 60)

    # Create environment
    env = KukaPickPlaceEnv(render_mode=None)
    teacher = IKTeacherWithEnv(env, gain=1.0)

    # Run a few episodes
    num_episodes = 5
    total_rewards = []
    successes = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = teacher.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        if info.get("distance_to_target", float("inf")) < 0.05:
            successes += 1

        print(
            f"Episode {episode + 1}: "
            f"Reward = {episode_reward:.2f}, "
            f"Distance = {info.get('distance_to_target', 'N/A'):.4f}, "
            f"Steps = {info.get('step_count', 'N/A')}"
        )

    env.close()

    print("-" * 60)
    print(f"Mean reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"Success rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    test_ik_teacher()
