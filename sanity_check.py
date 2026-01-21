#!/usr/bin/env python3
"""
sanity_check.py - Verify the KUKA mini-stacker environment works in Docker.

This script performs a series of checks to validate that all dependencies
and components are properly installed and functional. Run this after building
the Docker image to ensure the environment is correctly configured.

Usage:
    python sanity_check.py [--quick]

Options:
    --quick     Run only essential checks (skip training test)
"""

import argparse
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, Optional


# Check result container
@dataclass
class CheckResult:
    """Result of a single sanity check."""

    name: str
    passed: bool
    message: str
    details: Optional[str] = None


class SanityChecker:
    """Runs sanity checks for the KUKA environment."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: list[CheckResult] = []

    def run_check(
        self, name: str, check_fn: Callable[[], tuple[bool, str, Optional[str]]]
    ) -> CheckResult:
        """Run a single check and record the result."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"CHECK: {name}")
            print("=" * 60)

        try:
            passed, message, details = check_fn()
        except Exception as e:
            passed = False
            message = f"Exception: {type(e).__name__}: {e}"
            details = traceback.format_exc()

        result = CheckResult(name=name, passed=passed, message=message, details=details)
        self.results.append(result)

        if self.verbose:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {message}")
            if details and not passed:
                print(f"\nDetails:\n{details}")

        return result

    def print_summary(self) -> bool:
        """Print a summary of all check results. Returns True if all passed."""
        print("\n" + "=" * 60)
        print("SANITY CHECK SUMMARY")
        print("=" * 60)

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.name}: {result.message}")

        print("-" * 60)
        print(f"Results: {passed_count}/{total_count} checks passed")

        all_passed = passed_count == total_count
        if all_passed:
            print("\nAll sanity checks PASSED. Environment is ready!")
        else:
            print("\nSome checks FAILED. Please review the errors above.")

        print("=" * 60)
        return all_passed


def check_python_version() -> tuple[bool, str, Optional[str]]:
    """Check Python version meets minimum requirements."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    min_version = (3, 9)

    if version >= min_version:
        return True, f"Python {version_str} (>= 3.9 required)", None
    else:
        return (
            False,
            f"Python {version_str} is below minimum 3.9",
            f"Current: {version_str}, Required: >= 3.9",
        )


def check_core_dependencies() -> tuple[bool, str, Optional[str]]:
    """Check that core dependencies are installed."""
    required = {
        "numpy": "numpy",
        "gymnasium": "gymnasium",
        "pybullet": "pybullet",
    }
    missing = []
    versions = []

    for name, import_name in required.items():
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            versions.append(f"{name}=={version}")
        except ImportError:
            missing.append(name)

    if missing:
        return False, f"Missing packages: {', '.join(missing)}", None
    return True, f"Core deps OK: {', '.join(versions)}", None


def check_ml_dependencies() -> tuple[bool, str, Optional[str]]:
    """Check that ML/RL dependencies are installed."""
    required = {
        "torch": "torch",
        "stable_baselines3": "stable_baselines3",
    }
    missing = []
    versions = []

    for name, import_name in required.items():
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            versions.append(f"{name}=={version}")
        except ImportError:
            missing.append(name)

    if missing:
        return False, f"Missing packages: {', '.join(missing)}", None
    return True, f"ML deps OK: {', '.join(versions)}", None


def check_urdf_file() -> tuple[bool, str, Optional[str]]:
    """Check that the KUKA URDF file exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "kuka_3dof.urdf")

    if os.path.exists(urdf_path):
        size = os.path.getsize(urdf_path)
        return True, f"URDF found ({size} bytes)", urdf_path
    else:
        return False, "kuka_3dof.urdf not found", f"Expected at: {urdf_path}"


def check_mesh_assets() -> tuple[bool, str, Optional[str]]:
    """Check that PyBullet mesh assets are available."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mesh_dir = os.path.join(script_dir, "pybullet_kuka", "kuka_iiwa", "meshes")

    if os.path.isdir(mesh_dir):
        mesh_files = [
            f for f in os.listdir(mesh_dir) if f.endswith((".obj", ".stl", ".dae"))
        ]
        if mesh_files:
            return (
                True,
                f"Mesh assets found ({len(mesh_files)} files)",
                f"Directory: {mesh_dir}",
            )
        else:
            return (
                False,
                "Mesh directory exists but no mesh files found",
                f"Directory: {mesh_dir}",
            )
    else:
        # Check if PyBullet built-in data is available as fallback
        import pybullet_data

        pybullet_mesh_path = os.path.join(
            pybullet_data.getDataPath(), "kuka_iiwa", "meshes"
        )
        if os.path.isdir(pybullet_mesh_path):
            return (
                True,
                "Using PyBullet built-in KUKA meshes",
                f"Path: {pybullet_mesh_path}",
            )
        return (
            False,
            "Mesh assets not found",
            f"Expected at: {mesh_dir} or PyBullet data path",
        )


def check_pybullet_connection() -> tuple[bool, str, Optional[str]]:
    """Check that PyBullet can start in headless mode."""
    import pybullet as p

    try:
        physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -10)
        p.disconnect(physics_client)
        return True, "PyBullet DIRECT mode connection successful", None
    except Exception as e:
        return False, f"PyBullet connection failed: {e}", traceback.format_exc()


def check_environment_creation() -> tuple[bool, str, Optional[str]]:
    """Check that the KukaPickPlaceEnv can be instantiated."""
    from env_test import KukaPickPlaceEnv

    env = None
    try:
        env = KukaPickPlaceEnv(render_mode=None)
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        return (
            True,
            f"Environment created (obs: {obs_shape}, action: {action_shape})",
            None,
        )
    except Exception as e:
        return False, f"Environment creation failed: {e}", traceback.format_exc()
    finally:
        if env is not None:
            env.close()


def check_environment_reset() -> tuple[bool, str, Optional[str]]:
    """Check that the environment can reset."""
    from env_test import KukaPickPlaceEnv

    env = None
    try:
        env = KukaPickPlaceEnv(render_mode=None)
        obs, info = env.reset(seed=42)

        # Validate observation
        expected_obs_dim = 27  # Based on env_test.py
        if obs.shape[0] != expected_obs_dim:
            return (
                False,
                f"Observation dimension mismatch: {obs.shape[0]} != {expected_obs_dim}",
                None,
            )

        # Check info contains expected keys
        if "target_position" not in info:
            return False, "Info missing 'target_position' key", str(info.keys())

        return True, f"Reset successful, obs shape: {obs.shape}", None
    except Exception as e:
        return False, f"Environment reset failed: {e}", traceback.format_exc()
    finally:
        if env is not None:
            env.close()


def check_environment_step() -> tuple[bool, str, Optional[str]]:
    """Check that the environment can step."""
    from env_test import KukaPickPlaceEnv

    env = None
    try:
        env = KukaPickPlaceEnv(render_mode=None)
        obs, _ = env.reset(seed=42)

        # Take a few random steps
        total_reward = 0.0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Validate returns
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

        return True, f"10 steps completed, total reward: {total_reward:.3f}", None
    except Exception as e:
        return False, f"Environment step failed: {e}", traceback.format_exc()
    finally:
        if env is not None:
            env.close()


def check_rgb_rendering() -> tuple[bool, str, Optional[str]]:
    """Check that rgb_array rendering works."""
    from env_test import KukaPickPlaceEnv

    env = None
    try:
        env = KukaPickPlaceEnv(render_mode="rgb_array")
        env.reset(seed=42)

        # Take a step and render
        action = env.action_space.sample()
        env.step(action)
        rgb = env.render()

        if rgb is None:
            return False, "Render returned None", None

        # Check image dimensions
        height, width, channels = rgb.shape
        if channels != 3:
            return False, f"Expected 3 channels, got {channels}", None

        return True, f"RGB rendering OK ({width}x{height}x{channels})", None
    except Exception as e:
        return False, f"RGB rendering failed: {e}", traceback.format_exc()
    finally:
        if env is not None:
            env.close()


def check_sb3_model_creation() -> tuple[bool, str, Optional[str]]:
    """Check that a Stable Baselines3 PPO model can be created."""
    from stable_baselines3 import PPO
    from env_test import KukaPickPlaceEnv

    env = None
    try:
        env = KukaPickPlaceEnv(render_mode=None)
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            device="cpu",
            n_steps=64,  # Small for quick test
        )
        del model
        return True, "PPO model created successfully", None
    except Exception as e:
        return False, f"PPO model creation failed: {e}", traceback.format_exc()
    finally:
        if env is not None:
            env.close()


def check_training_step() -> tuple[bool, str, Optional[str]]:
    """Check that a short training run works."""
    from stable_baselines3 import PPO
    from env_test import KukaPickPlaceEnv

    env = None
    try:
        env = KukaPickPlaceEnv(render_mode=None)
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            device="cpu",
            n_steps=64,
            batch_size=32,
            n_epochs=1,
        )

        # Run a few training timesteps
        model.learn(total_timesteps=128, progress_bar=False)

        return True, "Training (128 timesteps) completed successfully", None
    except Exception as e:
        return False, f"Training failed: {e}", traceback.format_exc()
    finally:
        if env is not None:
            env.close()


def check_model_save_load() -> tuple[bool, str, Optional[str]]:
    """Check that model save/load works."""
    import tempfile
    from stable_baselines3 import PPO
    from env_test import KukaPickPlaceEnv

    env = None
    try:
        env = KukaPickPlaceEnv(render_mode=None)
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            device="cpu",
            n_steps=64,
        )

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_model")
            model.save(save_path)

            # Check file exists
            if not os.path.exists(save_path + ".zip"):
                return False, "Model file not created", None

            # Load and verify
            loaded_model = PPO.load(save_path, env=env, device="cpu")
            del loaded_model

        return True, "Model save/load successful", None
    except Exception as e:
        return False, f"Model save/load failed: {e}", traceback.format_exc()
    finally:
        if env is not None:
            env.close()


def run_sanity_checks(quick: bool = False) -> bool:
    """Run all sanity checks and return True if all pass."""
    checker = SanityChecker(verbose=True)

    print("\n" + "=" * 60)
    print("KUKA MINI-STACKER DOCKER SANITY CHECK")
    print("=" * 60)
    print(f"Mode: {'Quick' if quick else 'Full'}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python: {sys.executable}")

    # Essential checks
    checker.run_check("Python Version", check_python_version)
    checker.run_check("Core Dependencies", check_core_dependencies)
    checker.run_check("ML Dependencies", check_ml_dependencies)
    checker.run_check("URDF File", check_urdf_file)
    checker.run_check("Mesh Assets", check_mesh_assets)
    checker.run_check("PyBullet Connection", check_pybullet_connection)
    checker.run_check("Environment Creation", check_environment_creation)
    checker.run_check("Environment Reset", check_environment_reset)
    checker.run_check("Environment Step", check_environment_step)
    checker.run_check("RGB Rendering", check_rgb_rendering)

    if not quick:
        # Extended checks (may take longer)
        checker.run_check("SB3 Model Creation", check_sb3_model_creation)
        checker.run_check("Training Step", check_training_step)
        checker.run_check("Model Save/Load", check_model_save_load)

    return checker.print_summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sanity check for KUKA mini-stacker Docker environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only essential checks (skip training tests)",
    )
    args = parser.parse_args()

    all_passed = run_sanity_checks(quick=args.quick)

    if not all_passed:
        print("\nTo run this in Docker:")
        print("  docker build -t kuka-mini-stacker .")
        print("  docker run --rm kuka-mini-stacker python sanity_check.py")
        print("  docker run --rm kuka-mini-stacker python sanity_check.py --quick")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
