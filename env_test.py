"""
env_test.py - Truncated 4-DOF KUKA IIWA with gripper reaching environment.

MODERN DEEP RL STACK (2025):
This script demonstrates a fully modern implementation using:
1. Gymnasium: The modern API standard for Reinforcement Learning (replaces OpenAI Gym).
2. PyBullet: Used directly for physics simulation without legacy wrappers.
3. Custom Environment: A clean Gymnasium-compatible wrapper around PyBullet.

NO LEGACY DEPENDENCIES:
- No 'gym' library (deprecated since 2022)
- No 'pybullet_envs' package (legacy, incompatible with modern Gymnasium)
- Pure Gymnasium API with direct PyBullet integration

The robot uses a truncated 4-DOF KUKA IIWA arm with a prismatic gripper.
The URDF file (kuka_3dof.urdf) references meshes from pybullet_kuka/kuka_iiwa/meshes/.
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from typing import Optional, Tuple


# Constants
TARGET_FPS = 60.0  # Target frames per second for visual playback
SIMULATION_STEPS = 1000  # Number of simulation steps to run
MAX_EPISODE_STEPS = 150  # Maximum steps per episode

# KUKA-specific constants for the truncated 4-DOF KUKA + gripper reaching environment
KUKA_ARM_JOINT_COUNT = 4  # Truncated KUKA arm has 4 revolute joints
KUKA_ACTION_DIM = 5  # 4 arm joint deltas + 1 gripper action
KUKA_MAX_EPISODE_STEPS = 200
KUKA_EE_LINK_INDEX = 5  # Link index for body (end-effector) - link_0=base, joints 1-4, gripper_base fixed, body=5
KUKA_JOINT_DELTA_SCALE = 0.05  # Radians per step for arm joints
KUKA_GRIPPER_DELTA_SCALE = 0.01  # Meters per step for gripper

# Table configuration
TABLE_POSITION = [0.5, 0.0, 0.0]  # In front of the robot base
TABLE_HALF_EXTENTS = [0.3, 0.4, 0.3]  # Half-size: 0.6m x 0.8m x 0.6m tall
TABLE_HEIGHT = TABLE_HALF_EXTENTS[2] * 2  # Full height = 0.6m


class KukaPickPlaceEnv(gym.Env):
    """
    Truncated 4-DOF KUKA IIWA with gripper - reaching environment.

    This environment loads a truncated KUKA IIWA URDF (4 arm joints + prismatic
    gripper) with authentic KUKA mesh visuals. The task is reaching: move the
    gripper to a target position.

    Action space is 5-D:
    - First 4 components: arm joint position deltas
    - 5th component: gripper open/close (positive = open, negative = close)

    Observation space is 18-D:
    [4 arm_q, 4 arm_qdot, 2 gripper_q, 2 gripper_qdot, target_xyz, ee_xyz]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_episode_steps: int = KUKA_MAX_EPISODE_STEPS,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Action space: 4 arm joint deltas + 1 gripper action.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(KUKA_ACTION_DIM,),
            dtype=np.float32,
        )

        # Observation: [sin(q), cos(q), qdot, grip_pos, grip_vel,
        #               relative_target, ee_pos, prev_action] -> 27-D.
        # sin(q): 4, cos(q): 4, qdot: 4, grip_pos: 2, grip_vel: 2,
        # relative_target: 3, ee_pos: 3, prev_action: 5
        obs_dim = 4 + 4 + 4 + 2 + 2 + 3 + 3 + KUKA_ACTION_DIM  # 27
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Connect to PyBullet once and keep the connection across episodes.
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Physics configuration.
        p.setGravity(0, 0, -10.0)
        p.setTimeStep(1.0 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Path to the truncated 4-DOF KUKA URDF with gripper.
        # This URDF references meshes from pybullet_kuka/kuka_iiwa/meshes/
        self._repo_root = os.path.dirname(os.path.abspath(__file__))
        self._kuka_urdf_path = os.path.join(self._repo_root, "kuka_3dof.urdf")

        # Add repo root to PyBullet search path so it can resolve mesh paths
        p.setAdditionalSearchPath(self._repo_root)

        # Simulation handles
        self.plane_id = None
        self.robot_id = None
        self.table_id = None
        self.target_id = None

        # Joint indices will be populated in _setup_pybullet_kuka.
        self.arm_joint_indices = []
        self.gripper_joint_indices = []

        # Episode tracking
        self.step_count = 0
        self.target_pos = None
        self.last_end_effector_pos = None

        # Number of physics substeps per Gym step for smoother motion.
        self.sim_substeps_per_step = 4

    def _setup_pybullet_kuka(self):
        """
        Load the plane and the truncated 4-DOF KUKA IIWA robot with gripper.
        """
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Create table using a box primitive
        table_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=TABLE_HALF_EXTENTS
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=TABLE_HALF_EXTENTS,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],  # Wood-brown color
        )
        self.table_id = p.createMultiBody(
            baseMass=0.0,  # Static object
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[
                TABLE_POSITION[0],
                TABLE_POSITION[1],
                TABLE_HALF_EXTENTS[2],  # Raise so bottom touches ground
            ],
        )

        # Load the truncated 4-DOF KUKA URDF with gripper.
        if not os.path.exists(self._kuka_urdf_path):
            raise FileNotFoundError(
                f"KUKA URDF not found at {self._kuka_urdf_path}. "
                "Make sure kuka_3dof.urdf is present in the repo root."
            )
        self.robot_id = p.loadURDF(self._kuka_urdf_path, basePosition=[0, 0, 0])

        # Build joint index lists by querying joint names.
        arm_joint_names = [
            "lbr_iiwa_joint_1",
            "lbr_iiwa_joint_2",
            "lbr_iiwa_joint_3",
            "lbr_iiwa_joint_4",
        ]
        gripper_joint_names = [
            "left_finger_sliding_joint",
            "right_finger_sliding_joint",
        ]

        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        num_joints = p.getNumJoints(self.robot_id)
        joint_name_to_index = {}
        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            joint_name = joint_info[1].decode("utf-8")
            joint_name_to_index[joint_name] = joint_index

        for name in arm_joint_names:
            if name not in joint_name_to_index:
                raise KeyError(f"Expected arm joint {name} in KUKA URDF.")
            self.arm_joint_indices.append(joint_name_to_index[name])

        for name in gripper_joint_names:
            if name not in joint_name_to_index:
                raise KeyError(f"Expected gripper joint {name} in KUKA URDF.")
            self.gripper_joint_indices.append(joint_name_to_index[name])

        # Disable default motors to allow explicit position control.
        all_actuated_joints = self.arm_joint_indices + self.gripper_joint_indices
        p.setJointMotorControlArray(
            self.robot_id,
            all_actuated_joints,
            p.VELOCITY_CONTROL,
            forces=[0.0] * len(all_actuated_joints),
        )

        # Camera focusing on the arm (GUI only).
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=60,
                cameraPitch=-30,
                cameraTargetPosition=[0.0, 0.0, 0.4],
            )

    def _get_observation(self) -> np.ndarray:
        arm_states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        q = np.array([s[0] for s in arm_states], dtype=np.float32)
        qdot = np.array([s[1] for s in arm_states], dtype=np.float32)

        sin_q = np.sin(q)
        cos_q = np.cos(q)

        gripper_states = p.getJointStates(self.robot_id, self.gripper_joint_indices)
        grip_pos = np.array([s[0] for s in gripper_states], dtype=np.float32)
        grip_vel = np.array([s[1] for s in gripper_states], dtype=np.float32)

        ee_pos = np.array(
            p.getLinkState(
                self.robot_id, KUKA_EE_LINK_INDEX, computeForwardKinematics=True
            )[0],
            dtype=np.float32,
        )
        self.last_end_effector_pos = ee_pos

        relative_target = self.target_pos.astype(np.float32) - ee_pos

        obs = np.concatenate(
            [
                sin_q,
                cos_q,
                qdot,
                grip_pos,
                grip_vel,
                relative_target,
                ee_pos,
                self.prev_action,
            ]
        ).astype(np.float32)

        return obs

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the truncated 4-DOF KUKA with gripper reaching environment.
        """
        super().reset(seed=seed)

        # Clear simulation but keep connection.
        p.resetSimulation()

        # Reapply physics settings.
        p.setGravity(0, 0, -10.0)
        p.setTimeStep(1.0 / 240.0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Reload assets.
        self._setup_pybullet_kuka()

        # Initialize arm joints near zero with small random noise.
        arm_positions = self.np_random.uniform(
            low=-0.15 * np.pi, high=0.15 * np.pi, size=(KUKA_ARM_JOINT_COUNT,)
        )
        for idx, joint_idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, float(arm_positions[idx]), 0.0)

        # Initialize gripper in open position (left: 0, right: 0).
        for joint_idx in self.gripper_joint_indices:
            p.resetJointState(self.robot_id, joint_idx, 0.0, 0.0)

        # Randomize target position ON TOP of the table.
        x = self.np_random.uniform(
            TABLE_POSITION[0] - TABLE_HALF_EXTENTS[0] + 0.05,
            TABLE_POSITION[0] + TABLE_HALF_EXTENTS[0] - 0.05,
        )
        y = self.np_random.uniform(
            TABLE_POSITION[1] - TABLE_HALF_EXTENTS[1] + 0.05,
            TABLE_POSITION[1] + TABLE_HALF_EXTENTS[1] - 0.05,
        )
        z = TABLE_HEIGHT + 0.05  # Slightly above table surface (sphere radius)
        self.target_pos = np.array([x, y, z], dtype=np.float32)

        self.step_count = 0
        self.prev_distance = float("inf")  # initialize for progress reward
        self.prev_action = np.zeros(
            KUKA_ACTION_DIM, dtype=np.float32
        )  # for observation

        # Create a simple spherical target.
        target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.15, 0.6, 0.95, 1.0]
        )
        self.target_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=target_collision,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_pos.tolist(),
        )

        self.step_count = 0

        observation = self._get_observation()
        info = {"target_position": self.target_pos.copy()}
        return observation, info

    def _apply_action(self, action: np.ndarray):
        """
        Map the 5-D action into target joint positions.

        - First 4 elements: arm joint position deltas
        - 5th element: gripper action (positive = open, negative = close)
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Arm joint control: apply deltas to current positions.
        arm_deltas = action[:KUKA_ARM_JOINT_COUNT] * KUKA_JOINT_DELTA_SCALE

        arm_states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        arm_angles = np.array([s[0] for s in arm_states], dtype=np.float32)

        target_arm_angles = arm_angles + arm_deltas
        target_arm_angles = np.clip(target_arm_angles, -np.pi, np.pi)

        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_arm_angles.tolist(),
            forces=[300.0] * len(self.arm_joint_indices),
        )

        # Gripper control: 5th action controls opening/closing.
        # Positive action = open (fingers move apart), negative = close.
        gripper_action = action[4] * KUKA_GRIPPER_DELTA_SCALE

        gripper_states = p.getJointStates(self.robot_id, self.gripper_joint_indices)
        gripper_pos = np.array([s[0] for s in gripper_states], dtype=np.float32)

        # Left finger moves in negative x direction (limit: -0.055 to 0)
        # Right finger moves in positive x direction (limit: 0 to 0.055)
        # Opening = left goes more negative, right goes more positive
        target_left = np.clip(gripper_pos[0] - gripper_action, -0.055, 0.0)
        target_right = np.clip(gripper_pos[1] + gripper_action, 0.0, 0.055)

        p.setJointMotorControlArray(
            self.robot_id,
            self.gripper_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[target_left, target_right],
            forces=[80.0, 80.0],
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Run one timestep of the truncated 4-DOF KUKA with gripper reaching environment.
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._apply_action(action)

        # Store action for next observation
        self.prev_action = action.astype(np.float32)

        # Integrate physics for smoother motion.
        for _ in range(self.sim_substeps_per_step):
            p.stepSimulation()

        self.step_count += 1

        observation = self._get_observation()

        distance_to_target = float(
            np.linalg.norm(self.last_end_effector_pos - self.target_pos)
        )

        # shaping hyperparams (tunable)
        k_progress = 20.0  # reward for reducing distance this step
        k_distance = (
            1.0  # small penalty proportional to distance (keeps agent focusing)
        )
        k_action = 0.01  # penalty on action magnitude to encourage smoothness

        # compute progress reward
        if self.prev_distance == float("inf"):
            progress = 0.0
        else:
            progress = max(
                0.0, self.prev_distance - distance_to_target
            )  # only reward closing

        action_penalty = -k_action * float(np.sum(np.square(action)))

        reward = (
            k_progress * progress - k_distance * distance_to_target + action_penalty
        )

        # success
        if distance_to_target < 0.05:
            reward += 50.0
            terminated = True
        else:
            terminated = False

        # update prev_distance
        self.prev_distance = distance_to_target

        truncated = self.step_count >= self.max_episode_steps

        info = {
            "distance_to_target": distance_to_target,
            "step_count": self.step_count,
            "target_position": self.target_pos.copy(),
            "end_effector_position": self.last_end_effector_pos.copy(),
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            time.sleep(1.0 / TARGET_FPS)
            return None
        elif self.render_mode == "rgb_array":
            return self._get_camera_image()
        return None

    def _get_camera_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Capture an RGB image from a fixed camera viewpoint."""
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.4],
            distance=1.5,
            yaw=60,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width) / height,
            nearVal=0.1,
            farVal=100.0,
        )
        # Use TINY_RENDERER for headless compatibility (works without OpenGL display)
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        # Convert RGBA to RGB (drop alpha channel)
        rgb = np.array(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        return rgb

    def close(self):
        """Clean up and close the environment."""
        if getattr(self, "physics_client", None) is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


def main():
    """
    Main function for the truncated 4-DOF KUKA IIWA with gripper reaching environment.

    The KUKA robot will perform random actions to demonstrate that the physics
    simulation is working correctly.
    """
    print("=" * 60)
    print("Truncated 4-DOF KUKA IIWA with Gripper - Reaching Task")
    print("=" * 60)
    print("\nThis demo uses:")
    print("  ✓ Gymnasium (modern RL API)")
    print("  ✓ PyBullet (direct physics engine)")
    print("  ✓ Truncated KUKA IIWA mesh geometry (4 arm joints + gripper)")
    print("\n" + "=" * 60 + "\n")

    render_mode_env = os.environ.get("PYBULLET_RENDER_MODE", "human")
    if render_mode_env is not None and render_mode_env.lower() in (
        "none",
        "headless",
        "",
    ):
        resolved_render_mode = None
    else:
        resolved_render_mode = render_mode_env

    print(f"Creating KukaPickPlaceEnv with render_mode={resolved_render_mode!r}...")
    try:
        env = KukaPickPlaceEnv(render_mode=resolved_render_mode)
    except Exception as e:
        print(f"Error creating KukaPickPlaceEnv: {e}")
        print("\nTroubleshooting:")
        print("  1. Install PyBullet: pip install pybullet")
        print("  2. Install Gymnasium: pip install gymnasium")
        print(
            "  3. Ensure kuka_3dof.urdf and pybullet_kuka assets are present in the repo."
        )
        return

    print("KukaPickPlaceEnv created successfully!")
    print("\nInitializing environment...")

    try:
        observation, info = env.reset(seed=42)
        print(f"Environment reset. Observation shape: {observation.shape}")
        print(f"Target position: {info['target_position']}")
    except Exception as e:
        print(f"Error resetting KukaPickPlaceEnv: {e}")
        env.close()
        return

    print("\n" + "=" * 60)
    print("Truncated 4-DOF KUKA with Gripper - Simulation Started!")
    print("=" * 60)
    print("The 4-DOF KUKA arm with gripper will perform random reaching.")
    print("Watch the PyBullet window to see the real KUKA geometry and gripper.")
    print("\nPress Ctrl+C to stop the simulation.")
    print("=" * 60 + "\n")

    episode_count = 0

    try:
        for step in range(SIMULATION_STEPS):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            env.render()

            if terminated or truncated:
                episode_count += 1
                reason = "Target reached!" if terminated else "Time limit"
                print(
                    f"KUKA Episode {episode_count} finished (step {step}): {reason} "
                    f"(distance_to_target: {info['distance_to_target']:.3f})"
                )
                observation, info = env.reset()

    except KeyboardInterrupt:
        print("\n\nKUKA simulation interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nUnexpected error during KUKA simulation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nClosing KukaPickPlaceEnv and cleaning up resources...")
        env.close()
        print("KukaPickPlaceEnv closed successfully. Goodbye!")


if __name__ == "__main__":
    main()
