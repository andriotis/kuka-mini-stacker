"""
playground.py - Interactive mouse and keyboard control for KUKA robot and ball.

Controls:
- Left-click + drag: Move the robot's gripper (using inverse kinematics)
- Right-click + drag: Reposition the ball
- O key: Open gripper
- C key: Close gripper
- R key: Reset scene
- ESC or Q: Quit
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import math

# =============================================================================
# Constants
# =============================================================================

# Physics settings
SIMULATION_TIME_STEP = 1.0 / 240.0
TARGET_FPS = 60.0
PHYSICS_SUBSTEPS = 4

# Robot configuration
KUKA_EE_LINK_INDEX = 6  # End-effector link (link_7)
GRIPPER_LINK_INDEX = 7  # Gripper body link

# Table configuration
TABLE_POSITION = [0.5, 0.0, 0.0]
TABLE_HALF_EXTENTS = [0.3, 0.4, 0.3]
TABLE_HEIGHT = TABLE_HALF_EXTENTS[2] * 2

# Ball configuration
BALL_RADIUS = 0.04
BALL_MASS = 0.1

# Gripper limits
GRIPPER_OPEN_POS = 0.04
GRIPPER_CLOSE_POS = 0.0
GRIPPER_SPEED = 0.002

# Mouse button constants
MOUSE_LEFT_BUTTON = 0
MOUSE_MIDDLE_BUTTON = 1
MOUSE_RIGHT_BUTTON = 2

# Keyboard key constants (PyBullet key codes)
KEY_O = ord('o')
KEY_C = ord('c')
KEY_R = ord('r')
KEY_Q = ord('q')
KEY_ESCAPE = 27


# =============================================================================
# Scene Setup
# =============================================================================

class InteractivePlayground:
    """Interactive playground for controlling KUKA robot with mouse and keyboard."""

    def __init__(self):
        self.physics_client = None
        self.robot_id = None
        self.table_id = None
        self.ball_id = None
        self.plane_id = None

        # Joint indices
        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        self.movable_joint_indices = []
        self.joint_idx_to_ik_output_idx = {}

        # Gripper state
        self.gripper_target = GRIPPER_CLOSE_POS

        # Mouse state
        self.dragging_gripper = False
        self.dragging_ball = False
        self.target_gripper_pos = None

        # Paths
        self._repo_root = os.path.dirname(os.path.abspath(__file__))
        self._kuka_urdf_path = os.path.join(self._repo_root, "kuka_3dof.urdf")

    def setup(self):
        """Initialize PyBullet and load the scene."""
        # Connect to PyBullet GUI
        self.physics_client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        # Physics settings
        p.setGravity(0, 0, -10.0)
        p.setTimeStep(SIMULATION_TIME_STEP)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(self._repo_root)

        # Load scene
        self._load_scene()

        # Setup camera
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=60,
            cameraPitch=-30,
            cameraTargetPosition=[0.3, 0.0, 0.4],
        )

        # Get initial end-effector position
        ee_state = p.getLinkState(self.robot_id, KUKA_EE_LINK_INDEX)
        self.target_gripper_pos = np.array(ee_state[0])

        print("\n" + "=" * 60)
        print("KUKA Interactive Playground")
        print("=" * 60)
        print("\nControls:")
        print("  Left-click + drag  : Move gripper")
        print("  Right-click + drag : Move ball")
        print("  O                  : Open gripper")
        print("  C                  : Close gripper")
        print("  R                  : Reset scene")
        print("  Q / ESC            : Quit")
        print("=" * 60 + "\n")

    def _load_scene(self):
        """Load ground plane, table, robot, and ball."""
        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Table
        table_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=TABLE_HALF_EXTENTS
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=TABLE_HALF_EXTENTS,
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
        )
        self.table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[
                TABLE_POSITION[0],
                TABLE_POSITION[1],
                TABLE_HALF_EXTENTS[2],
            ],
        )

        # Robot
        self.robot_id = p.loadURDF(self._kuka_urdf_path, basePosition=[0, 0, 0])
        self._setup_robot_joints()

        # Ball
        self._create_ball()

    def _setup_robot_joints(self):
        """Configure robot joint indices and disable default motors."""
        arm_joint_names = [
            "lbr_iiwa_joint_1",
            "lbr_iiwa_joint_2",
            "lbr_iiwa_joint_3",
            "lbr_iiwa_joint_4",
            "lbr_iiwa_joint_5",
            "lbr_iiwa_joint_6",
            "lbr_iiwa_joint_7",
        ]
        gripper_joint_names = [
            "left_finger_sliding_joint",
            "right_finger_sliding_joint",
        ]

        num_joints = p.getNumJoints(self.robot_id)
        joint_name_to_index = {}

        for joint_index in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_index)
            joint_name = joint_info[1].decode("utf-8")
            joint_name_to_index[joint_name] = joint_index

        # Map arm joints
        for name in arm_joint_names:
            if name in joint_name_to_index:
                self.arm_joint_indices.append(joint_name_to_index[name])

        # Map gripper joints
        for name in gripper_joint_names:
            if name in joint_name_to_index:
                self.gripper_joint_indices.append(joint_name_to_index[name])

        # Build movable joint list for IK
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            joint_type = joint_info[2]
            if joint_type != p.JOINT_FIXED:
                self.movable_joint_indices.append(joint_idx)

        self.joint_idx_to_ik_output_idx = {
            joint_idx: i for i, joint_idx in enumerate(self.movable_joint_indices)
        }

        # Disable default motors
        all_actuated = self.arm_joint_indices + self.gripper_joint_indices
        p.setJointMotorControlArray(
            self.robot_id,
            all_actuated,
            p.VELOCITY_CONTROL,
            forces=[0.0] * len(all_actuated),
        )

        # Set initial arm pose (slightly bent)
        initial_positions = [0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0]
        for idx, joint_idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, initial_positions[idx], 0.0)

    def _create_ball(self):
        """Create a ball on the table."""
        ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS)
        ball_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=BALL_RADIUS,
            rgbaColor=[0.9, 0.2, 0.2, 1.0],
        )
        ball_pos = [
            TABLE_POSITION[0],
            TABLE_POSITION[1],
            TABLE_HEIGHT + BALL_RADIUS + 0.01,
        ]
        self.ball_id = p.createMultiBody(
            baseMass=BALL_MASS,
            baseCollisionShapeIndex=ball_collision,
            baseVisualShapeIndex=ball_visual,
            basePosition=ball_pos,
        )

    def reset_scene(self):
        """Reset robot pose and ball position."""
        # Reset arm to initial pose
        initial_positions = [0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0]
        for idx, joint_idx in enumerate(self.arm_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, initial_positions[idx], 0.0)

        # Reset gripper
        for joint_idx in self.gripper_joint_indices:
            p.resetJointState(self.robot_id, joint_idx, 0.0, 0.0)
        self.gripper_target = GRIPPER_CLOSE_POS

        # Reset ball position
        ball_pos = [
            TABLE_POSITION[0],
            TABLE_POSITION[1],
            TABLE_HEIGHT + BALL_RADIUS + 0.01,
        ]
        p.resetBasePositionAndOrientation(
            self.ball_id, ball_pos, [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0])

        # Update target gripper position
        ee_state = p.getLinkState(self.robot_id, KUKA_EE_LINK_INDEX)
        self.target_gripper_pos = np.array(ee_state[0])

        print("Scene reset.")

    # =========================================================================
    # Mouse Interaction
    # =========================================================================

    def screen_to_world_ray(self, mouse_x, mouse_y):
        """Convert screen coordinates to a 3D world ray."""
        # Get camera info
        cam_info = p.getDebugVisualizerCamera()
        width = cam_info[0]
        height = cam_info[1]
        view_matrix = np.array(cam_info[2]).reshape(4, 4, order='F')
        proj_matrix = np.array(cam_info[3]).reshape(4, 4, order='F')

        # Normalize device coordinates
        ndc_x = (2.0 * mouse_x / width) - 1.0
        ndc_y = 1.0 - (2.0 * mouse_y / height)

        # Inverse projection matrix
        proj_inv = np.linalg.inv(proj_matrix)
        view_inv = np.linalg.inv(view_matrix)

        # Near and far points in NDC
        near_ndc = np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_ndc = np.array([ndc_x, ndc_y, 1.0, 1.0])

        # Transform to view space
        near_view = proj_inv @ near_ndc
        far_view = proj_inv @ far_ndc
        near_view /= near_view[3]
        far_view /= far_view[3]

        # Transform to world space
        near_world = view_inv @ near_view
        far_world = view_inv @ far_view
        near_world /= near_world[3]
        far_world /= far_world[3]

        ray_origin = near_world[:3]
        ray_direction = far_world[:3] - near_world[:3]
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        return ray_origin, ray_direction

    def ray_plane_intersection(self, ray_origin, ray_direction, plane_point, plane_normal):
        """Find intersection of ray with a plane."""
        denom = np.dot(ray_direction, plane_normal)
        if abs(denom) < 1e-6:
            return None

        t = np.dot(plane_point - ray_origin, plane_normal) / denom
        if t < 0:
            return None

        return ray_origin + t * ray_direction

    def get_mouse_world_position(self, mouse_x, mouse_y, reference_height=None):
        """Get 3D world position from mouse coordinates at a given height."""
        ray_origin, ray_direction = self.screen_to_world_ray(mouse_x, mouse_y)

        # Use current gripper height or table height as reference
        if reference_height is None:
            ee_state = p.getLinkState(self.robot_id, KUKA_EE_LINK_INDEX)
            reference_height = ee_state[0][2]

        # Intersect with horizontal plane at reference height
        plane_point = np.array([0, 0, reference_height])
        plane_normal = np.array([0, 0, 1])

        intersection = self.ray_plane_intersection(
            ray_origin, ray_direction, plane_point, plane_normal
        )

        return intersection

    # =========================================================================
    # Input Handling
    # =========================================================================

    def handle_keyboard(self):
        """Process keyboard input."""
        keys = p.getKeyboardEvents()

        # Quit
        if KEY_Q in keys or KEY_ESCAPE in keys:
            return False

        # Gripper control
        if KEY_O in keys and keys[KEY_O] & p.KEY_WAS_TRIGGERED:
            self.gripper_target = GRIPPER_OPEN_POS
            print("Gripper: OPEN")

        if KEY_C in keys and keys[KEY_C] & p.KEY_WAS_TRIGGERED:
            self.gripper_target = GRIPPER_CLOSE_POS
            print("Gripper: CLOSE")

        # Reset
        if KEY_R in keys and keys[KEY_R] & p.KEY_WAS_TRIGGERED:
            self.reset_scene()

        return True

    def handle_mouse(self):
        """Process mouse input for dragging gripper and ball."""
        mouse_events = p.getMouseEvents()

        for event in mouse_events:
            event_type, mouse_x, mouse_y, button_idx, button_state = event

            # Left mouse button - control gripper
            if button_idx == MOUSE_LEFT_BUTTON:
                if button_state & p.KEY_IS_DOWN:
                    self.dragging_gripper = True
                    world_pos = self.get_mouse_world_position(mouse_x, mouse_y)
                    if world_pos is not None:
                        # Clamp to reasonable workspace
                        world_pos[0] = np.clip(world_pos[0], -0.2, 0.9)
                        world_pos[1] = np.clip(world_pos[1], -0.6, 0.6)
                        world_pos[2] = np.clip(world_pos[2], 0.1, 1.2)
                        self.target_gripper_pos = world_pos
                elif button_state & p.KEY_WAS_RELEASED:
                    self.dragging_gripper = False

            # Right mouse button - move ball
            if button_idx == MOUSE_RIGHT_BUTTON:
                if button_state & p.KEY_IS_DOWN:
                    self.dragging_ball = True
                    world_pos = self.get_mouse_world_position(
                        mouse_x, mouse_y,
                        reference_height=TABLE_HEIGHT + BALL_RADIUS + 0.01
                    )
                    if world_pos is not None:
                        # Clamp ball to table surface area
                        world_pos[0] = np.clip(
                            world_pos[0],
                            TABLE_POSITION[0] - TABLE_HALF_EXTENTS[0] + BALL_RADIUS,
                            TABLE_POSITION[0] + TABLE_HALF_EXTENTS[0] - BALL_RADIUS,
                        )
                        world_pos[1] = np.clip(
                            world_pos[1],
                            TABLE_POSITION[1] - TABLE_HALF_EXTENTS[1] + BALL_RADIUS,
                            TABLE_POSITION[1] + TABLE_HALF_EXTENTS[1] - BALL_RADIUS,
                        )
                        world_pos[2] = TABLE_HEIGHT + BALL_RADIUS + 0.01
                        p.resetBasePositionAndOrientation(
                            self.ball_id, world_pos.tolist(), [0, 0, 0, 1]
                        )
                        p.resetBaseVelocity(self.ball_id, [0, 0, 0], [0, 0, 0])
                elif button_state & p.KEY_WAS_RELEASED:
                    self.dragging_ball = False

    # =========================================================================
    # Robot Control
    # =========================================================================

    def compute_ik(self, target_position):
        """Compute joint angles to reach target end-effector position."""
        # Build IK parameters
        rest_poses = []
        lower_limits = []
        upper_limits = []
        joint_ranges = []
        joint_damping = []

        for joint_idx in self.movable_joint_indices:
            rest_poses.append(p.getJointState(self.robot_id, joint_idx)[0])
            info = p.getJointInfo(self.robot_id, joint_idx)
            lower = info[8]
            upper = info[9]
            lower_limits.append(lower)
            upper_limits.append(upper)
            if lower < upper:
                joint_ranges.append(upper - lower)
            else:
                joint_ranges.append(2.0 * np.pi)
            joint_damping.append(0.1)

        ik_solution = p.calculateInverseKinematics(
            self.robot_id,
            KUKA_EE_LINK_INDEX,
            target_position.tolist(),
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            jointDamping=joint_damping,
            maxNumIterations=100,
            residualThreshold=1e-4,
        )

        # Extract arm joint values
        arm_angles = np.array([
            ik_solution[self.joint_idx_to_ik_output_idx[idx]]
            for idx in self.arm_joint_indices
        ])

        return arm_angles

    def update_robot(self):
        """Update robot arm and gripper positions."""
        # Compute IK for target position
        target_angles = self.compute_ik(self.target_gripper_pos)

        # Apply arm position control
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_angles.tolist(),
            forces=[300.0] * len(self.arm_joint_indices),
        )

        # Update gripper
        # Left finger: negative direction, Right finger: positive direction
        target_left = -self.gripper_target
        target_right = self.gripper_target

        p.setJointMotorControlArray(
            self.robot_id,
            self.gripper_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[target_left, target_right],
            forces=[80.0, 80.0],
        )

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run(self):
        """Main simulation loop."""
        self.setup()

        running = True
        last_time = time.time()

        try:
            while running:
                # Handle input
                running = self.handle_keyboard()
                if not running:
                    break

                self.handle_mouse()

                # Update robot
                self.update_robot()

                # Step physics
                for _ in range(PHYSICS_SUBSTEPS):
                    p.stepSimulation()

                # Frame rate control
                current_time = time.time()
                elapsed = current_time - last_time
                sleep_time = (1.0 / TARGET_FPS) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.close()

    def close(self):
        """Clean up and disconnect."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
        print("Playground closed. Goodbye!")


def main():
    """Entry point."""
    playground = InteractivePlayground()
    playground.run()


if __name__ == "__main__":
    main()
