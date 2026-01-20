"""
env_test.py - A modern PyBullet + Gymnasium robotics simulation.

MODERN DEEP RL STACK (2025):
This script demonstrates a fully modern implementation using:
1. Gymnasium: The modern API standard for Reinforcement Learning (replaces OpenAI Gym).
2. PyBullet: Used directly for physics simulation without legacy wrappers.
3. Custom Environment: A clean Gymnasium-compatible wrapper around PyBullet.

NO LEGACY DEPENDENCIES:
- No 'gym' library (deprecated since 2022)
- No 'pybullet_envs' package (legacy, incompatible with modern Gymnasium)
- Pure Gymnasium API with direct PyBullet integration
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


class PyBulletReacherEnv(gym.Env):
    """
    A modern Gymnasium environment for a 2-joint robot arm reaching task.

    This environment wraps PyBullet directly, demonstrating how to create
    a custom Gymnasium environment without legacy dependencies.

    The task: A 2-joint arm tries to reach a randomly positioned target.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the PyBullet Reacher environment.

        Args:
            render_mode: The render mode to use. Options: "human", "rgb_array", or None.
        """
        super().__init__()

        self.render_mode = render_mode

        # Action space: torques for 3 joints (continuous)
        # Each joint can apply torque in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space (3D): [q1, q1_dot, q2, q2_dot, q3, q3_dot,
        #                          target_x, target_y, target_z, ee_x, ee_y, ee_z]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        # Connect to PyBullet once (keep connection alive for all episodes)
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable GUI controls
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Set up physics engine parameters (these persist across resets)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)  # 240 Hz physics update rate
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Initialize simulation objects (will be set in reset)
        self.robot_id = None
        self.target_id = None
        self.plane_id = None

        # Episode tracking
        self.step_count = 0
        self.target_pos = None
        self.last_end_effector_pos = None
        self.arm_joint_indices = [0, 1, 2]  # Will be set in _setup_pybullet

    def _setup_pybullet(self):
        """Load simulation assets (robot, target, plane) into PyBullet."""
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Define robot via URDF string (The reliable industry standard)
        # This is much cleaner than createMultiBody for complex chains.
        urdf_str = """
        <?xml version="1.0"?>
        <robot name="reacher">
          <link name="base_link">
            <!-- Visual-only: industrial pedestal base with accent top plate -->
            <visual>
              <geometry><box size="0.18 0.18 0.06"/></geometry>
              <material name="base_dark"><color rgba="0.15 0.15 0.18 1"/></material>
            </visual>
            <visual>
              <origin xyz="0 0 0.05"/>
              <geometry><box size="0.12 0.12 0.02"/></geometry>
              <material name="base_accent"><color rgba="0.85 0.85 0.88 1"/></material>
            </visual>
            <collision>
              <!-- Collision kept as original 0.1 m cube to preserve physics -->
              <geometry><box size="0.1 0.1 0.1"/></geometry>
            </collision>
            <inertial>
              <mass value="0"/> <!-- 0 mass means fixed to world -->
              <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
            </inertial>
          </link>
          
          <joint name="joint1" type="revolute">
            <parent link="base_link"/>
            <child link="link1"/>
            <origin xyz="0 0 0.05"/> <!-- At the top of the base box -->
            <axis xyz="0 0 1"/>
            <limit effort="10" velocity="10" lower="-3.14" upper="3.14"/>
          </joint>
          
          <link name="link1">
            <!-- Upper arm: oriented vertically so it starts perpendicular to the base -->
            <visual>
              <origin xyz="0 0 0.15"/> <!-- Center of 0.3m link along Z -->
              <geometry><box size="0.05 0.05 0.3"/></geometry>
              <material name="arm_upper"><color rgba="0.9 0.7 0.15 1"/></material>
            </visual>
            <!-- Base-side joint housing -->
            <visual>
              <origin xyz="0 0 0.0"/>
              <geometry><cylinder length="0.055" radius="0.055"/></geometry>
              <material name="joint_housing"><color rgba="0.3 0.3 0.35 1"/></material>
            </visual>
            <!-- Elbow-side joint flange -->
            <visual>
              <origin xyz="0 0 0.3"/>
              <geometry><cylinder length="0.04" radius="0.045"/></geometry>
              <material name="joint_flange"><color rgba="0.4 0.4 0.45 1"/></material>
            </visual>
            <collision>
              <origin xyz="0 0 0.15"/>
              <geometry><box size="0.05 0.05 0.3"/></geometry>
            </collision>
            <inertial>
              <mass value="1"/>
              <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
            </inertial>
          </link>
          
          <joint name="joint2" type="revolute">
            <parent link="link1"/>
            <child link="link2"/>
            <origin xyz="0 0 0.3"/> <!-- At the top of vertical link1 -->
            <axis xyz="0 1 0"/> <!-- Shoulder pitch -->
            <limit effort="10" velocity="10" lower="-3.14" upper="3.14"/>
          </joint>
          
          <link name="link2">
            <!-- Visual-only: forearm with end-effector module -->
            <visual>
              <origin xyz="0.125 0 0"/> <!-- Center of 0.25m link along X -->
              <geometry><box size="0.25 0.04 0.04"/></geometry>
              <material name="arm_lower"><color rgba="0.95 0.8 0.2 1"/></material>
            </visual>
            <!-- Elbow joint housing -->
            <visual>
              <origin xyz="0 0 0.0"/>
              <geometry><cylinder length="0.045" radius="0.045"/></geometry>
              <material name="joint_housing"><color rgba="0.3 0.3 0.35 1"/></material>
            </visual>
            <!-- End-effector mounting plate -->
            <visual>
              <origin xyz="0.25 0 0"/>
              <geometry><box size="0.04 0.06 0.02"/></geometry>
              <material name="ee_plate"><color rgba="0.2 0.3 0.6 1"/></material>
            </visual>
            <!-- End-effector tip for screenshots (visual only, tiny) -->
            <visual>
              <origin xyz="0.27 0 0"/>
              <geometry><sphere radius="0.025"/></geometry>
              <material name="ee_tip"><color rgba="0.95 0.25 0.25 1"/></material>
            </visual>
            <collision>
              <!-- Collision kept as original box to preserve dynamics -->
              <origin xyz="0.125 0 0"/>
              <geometry><box size="0.25 0.04 0.04"/></geometry>
            </collision>
            <inertial>
              <mass value="1"/>
              <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
            </inertial>
          </link>

          <joint name="joint3" type="revolute">
            <parent link="link2"/>
            <child link="link3"/>
            <origin xyz="0.25 0 0"/> <!-- At the end of link2 -->
            <axis xyz="0 1 0"/> <!-- Elbow pitch -->
            <limit effort="10" velocity="10" lower="-3.14" upper="3.14"/>
          </joint>

          <link name="link3">
            <!-- Short wrist/end-effector stub -->
            <visual>
              <origin xyz="0.06 0 0"/>
              <geometry><box size="0.12 0.03 0.03"/></geometry>
              <material name="ee_body"><color rgba="0.85 0.85 0.9 1"/></material>
            </visual>
            <visual>
              <origin xyz="0.12 0 0"/>
              <geometry><sphere radius="0.02"/></geometry>
              <material name="ee_tip"><color rgba="0.95 0.25 0.25 1"/></material>
            </visual>
            <collision>
              <origin xyz="0.06 0 0"/>
              <geometry><box size="0.12 0.03 0.03"/></geometry>
            </collision>
            <inertial>
              <mass value="0.2"/>
              <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
            </inertial>
          </link>
        </robot>
        """

        # Write URDF to temporary file and load it
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(urdf_str)
            urdf_path = f.name

        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
        os.unlink(urdf_path)

        # In URDF, revolute joints are automatically indexed starting from 0
        self.arm_joint_indices = [0, 1, 2]

        # Verify joints
        num_joints = p.getNumJoints(self.robot_id)
        if num_joints < 3:
            raise RuntimeError(f"Expected 3 joints but found {num_joints}")

        # Disable default motors to enable torque control
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            p.VELOCITY_CONTROL,
            forces=[0, 0, 0],
        )

        # Create target sphere (keep collision radius, make visual slightly larger/brighter)
        target_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.06, rgbaColor=[0.15, 0.6, 0.95, 1.0]
        )
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=target_collision,
            baseVisualShapeIndex=target_visual,
            basePosition=[0.5, 0, 0.3],
        )

        # Set a nicer default camera view for report-quality screenshots (GUI only)
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.4,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[0.35, 0.0, 0.2],
            )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state.

        Modern Gymnasium standard: Returns (observation, info).

        Args:
            seed: Optional random seed for reproducibility
            options: Optional dictionary with additional reset options

        Returns:
            observation: Initial observation of the environment
            info: Dictionary with additional information
        """
        super().reset(seed=seed)

        # Reset simulation without disconnecting (keeps GUI window open)
        # This is much faster and more stable than disconnecting/reconnecting
        p.resetSimulation()

        # Restore physics settings (resetSimulation clears them)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        # Reload simulation assets
        self._setup_pybullet()

        # Reset robot arm joints.
        # Start upright (link1 is vertical in the URDF) and add a small random perturbation.
        joint_positions = self.np_random.uniform(
            low=-0.15 * np.pi, high=0.15 * np.pi, size=(3,)
        )
        for idx, joint_idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, joint_positions[idx], 0.0)

        # Randomize target position in 3D within a reachable shell.
        # Keep it above the ground plane for better visuals and fewer collisions.
        azimuth = self.np_random.uniform(0, 2 * np.pi)
        elevation = self.np_random.uniform(-0.35 * np.pi, 0.25 * np.pi)
        radius = self.np_random.uniform(0.25, 0.55)
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = max(0.08, radius * np.sin(elevation) + 0.25)
        self.target_pos = np.array([x, y, z], dtype=np.float32)
        p.resetBasePositionAndOrientation(
            self.target_id, self.target_pos.tolist(), [0, 0, 0, 1]
        )

        # Reset episode counter
        self.step_count = 0

        # Get initial observation
        observation = self._get_observation()

        info = {"target_position": self.target_pos.copy()}

        return observation, info

    def _get_observation(self) -> np.ndarray:
        """Compute the current observation vector."""
        # Get joint states for revolute arm joints
        joint_states = p.getJointStates(self.robot_id, self.arm_joint_indices)
        joint_angles = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        # End-effector position from PyBullet forward kinematics (world frame).
        # The end-effector is the final link attached to joint3.
        end_effector_world_pos = np.array(
            p.getLinkState(self.robot_id, linkIndex=2, computeForwardKinematics=True)[
                0
            ],
            dtype=np.float32,
        )
        self.last_end_effector_pos = end_effector_world_pos

        # Construct observation (3D):
        # [q1, q1_dot, q2, q2_dot, q3, q3_dot, target_xyz, ee_xyz]
        observation = np.array(
            [
                joint_angles[0],
                joint_velocities[0],
                joint_angles[1],
                joint_velocities[1],
                joint_angles[2],
                joint_velocities[2],
                self.target_pos[0],
                self.target_pos[1],
                self.target_pos[2],
                end_effector_world_pos[0],
                end_effector_world_pos[1],
                end_effector_world_pos[2],
            ],
            dtype=np.float32,
        )

        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Run one timestep of the environment's dynamics.

        Modern Gymnasium standard: Returns 5 values.
        (observation, reward, terminated, truncated, info)

        Args:
            action: The action to take (torques for 3 joints)

        Returns:
            observation: Observation of the environment after the step
            reward: Reward for this step
            terminated: Whether the episode ended due to task completion
            truncated: Whether the episode ended due to time limit
            info: Dictionary with additional information
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply torques to arm joints (indices 1 and 2)
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,  # Use arm joint indices, not 0 and 1
            p.TORQUE_CONTROL,
            forces=action * 2.0,  # Scale action to reasonable torque values
        )

        # Step simulation
        p.stepSimulation()

        # Increment step counter
        self.step_count += 1

        # Get new observation
        observation = self._get_observation()

        # Calculate reward: negative 3D distance to target
        distance_to_target = float(
            np.linalg.norm(self.last_end_effector_pos - self.target_pos)
        )
        reward = -distance_to_target

        # Bonus for reaching target (within 5cm)
        if distance_to_target < 0.05:
            reward += 10.0
            terminated = True
        else:
            terminated = False

        # Episode truncation: time limit reached
        truncated = self.step_count >= MAX_EPISODE_STEPS

        info = {
            "distance_to_target": distance_to_target,
            "step_count": self.step_count,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment (handled automatically by PyBullet GUI)."""
        if self.render_mode == "human":
            # PyBullet GUI handles rendering automatically
            time.sleep(1.0 / TARGET_FPS)  # Control playback speed
        elif self.render_mode == "rgb_array":
            # Could implement camera-based rendering here if needed
            pass

    def close(self):
        """Clean up and close the environment."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


def main():
    """
    Main function that demonstrates the modern PyBullet + Gymnasium environment.

    The robot arm will perform random actions to demonstrate that the physics
    simulation is working correctly.
    """
    print("=" * 60)
    print("Modern PyBullet + Gymnasium Robot Simulation")
    print("=" * 60)
    print("\nThis script uses:")
    print("  ✓ Gymnasium (modern RL API)")
    print("  ✓ PyBullet (direct physics engine)")
    print("  ✓ Custom Gymnasium environment (no legacy dependencies)")
    print("\n" + "=" * 60 + "\n")

    # Create the environment, choosing render mode based on environment variables.
    # Default is "human" (GUI) locally; in headless/docker we can set PYBULLET_RENDER_MODE.
    render_mode_env = os.environ.get("PYBULLET_RENDER_MODE", "human")
    if render_mode_env is not None and render_mode_env.lower() in (
        "none",
        "headless",
        "",
    ):
        resolved_render_mode = None
    else:
        resolved_render_mode = render_mode_env

    print(f"Creating PyBulletReacherEnv with render_mode={resolved_render_mode!r}...")
    try:
        env = PyBulletReacherEnv(render_mode=resolved_render_mode)
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nTroubleshooting:")
        print("  1. Install PyBullet: pip install pybullet")
        print("  2. Install Gymnasium: pip install gymnasium")
        print("  3. For headless servers: use render_mode=None or xvfb")
        return

    print("Environment created successfully!")
    print("\nInitializing environment...")

    # Reset the environment (modern Gymnasium: returns obs, info)
    try:
        observation, info = env.reset(seed=42)
        print(f"Environment reset. Observation shape: {observation.shape}")
        print(f"Target position: {info['target_position']}")
    except Exception as e:
        print(f"Error resetting environment: {e}")
        env.close()
        return

    print("\n" + "=" * 60)
    print("Simulation Started!")
    print("=" * 60)
    print("The robot arm will perform random movements.")
    print("Watch the PyBullet window to see the robot in action.")
    print("\nPress Ctrl+C to stop the simulation.")
    print("=" * 60 + "\n")

    episode_count = 0

    try:
        for step in range(SIMULATION_STEPS):
            # Sample a random action from the action space
            action = env.action_space.sample()

            # Step the environment (modern Gymnasium: returns 5 values)
            observation, reward, terminated, truncated, info = env.step(action)

            # Render (controls playback speed)
            env.render()

            # Handle episode termination
            if terminated or truncated:
                episode_count += 1
                reason = "Target reached!" if terminated else "Time limit"
                print(
                    f"Episode {episode_count} finished (step {step}): {reason} "
                    f"(distance: {info['distance_to_target']:.3f})"
                )
                observation, info = env.reset()

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user (Ctrl+C).")

    except Exception as e:
        print(f"\nUnexpected error during simulation: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up (always executed)
        print("\nClosing environment and cleaning up resources...")
        env.close()
        print("Environment closed successfully. Goodbye!")


if __name__ == "__main__":
    main()
