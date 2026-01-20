## KUKA Mini Stacker – Modern PyBullet + Gymnasium Demo

This repository contains a **modern reinforcement learning–ready robot arm simulation** built with **PyBullet** and **Gymnasium**.  
It implements a custom `PyBulletReacherEnv` where a 3-DOF arm reaches randomly placed 3D targets, designed as a clean, up-to-date example of how to:

- **Integrate PyBullet directly** (no legacy `pybullet_envs`)
- **Use the Gymnasium API** (no deprecated `gym` dependency)
- **Package a simulation for local and Docker/headless usage**

### Project Status

- **Maturity**: Prototype / demo
- **Core environment**: Implemented and working (random policy rollout)
- **RL training code**: **Not yet included** (environment is RL-ready)
- **Headless support**: Docker image runs PyBullet in DIRECT/headless mode via `PYBULLET_RENDER_MODE=headless`

### Features

- **Custom Gymnasium environment** (`PyBulletReacherEnv`) wrapping a PyBullet simulation
- **3-joint robot arm** defined via an inline URDF with visually clean geometry
- **Random 3D target placement** within a reachable workspace
- **Shaped reward** based on distance to the target, with a success bonus
- **Modern Gymnasium API**: `(obs, info)` on reset and 5-tuple return from `step`
- **Dockerfile** for reproducible, headless runs using `uv` for dependency installation

---

### Getting Started (Local)

1. **Create and activate a virtualenv** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the simulation** (GUI by default, if available):

```bash
python env_test.py
```

If you are on a headless machine or want to disable the GUI, set:

```bash
export PYBULLET_RENDER_MODE=headless
python env_test.py
```

---

### Running with Docker

Build the image:

```bash
docker build -t kuka-mini-stacker .
```

Run the container (headless by default via `PYBULLET_RENDER_MODE=headless`):

```bash
docker run --rm kuka-mini-stacker
```

If you want to experiment with other render modes, override the env variable:

```bash
docker run --rm -e PYBULLET_RENDER_MODE=none kuka-mini-stacker
```

> Note: For full GUI rendering from inside Docker, additional host/Display/X11 configuration would be required and is not set up by default.

---

### Code Layout

- `env_test.py` – Main script containing:
  - `PyBulletReacherEnv`: Gymnasium-compatible environment for the 3-DOF arm
  - `main()`: Demo loop that samples random actions and runs the simulation
- `requirements.txt` – Minimal set of Python dependencies (Gymnasium, PyBullet, NumPy)
- `Dockerfile` – Container image definition using Python 3.11 and `uv` for installing dependencies

---

### Screenshots

_Placeholders for future screenshots of the simulation:_

- **Screenshot 1**: Arm in default pose with target sphere visible.
- **Screenshot 2**: Arm successfully reaching a nearby target.
- **Screenshot 3**: Example of multiple targets over different episodes (collage).

Once you capture images, you can embed them here, for example:

```markdown
![KUKA Mini Stacker – Initial Pose](docs/images/initial_pose.png)
![KUKA Mini Stacker – Reaching Target](docs/images/reach_target.png)
```

---

### Next Steps / Ideas

- Add **training scripts** with a modern RL library (e.g., Stable-Baselines3, CleanRL, or custom PPO/SAC) to learn a policy on `PyBulletReacherEnv`.
- Extend the environment with **obstacles**, **different reward structures**, or **partial observability**.
- Integrate **logging and video recording** for automated experiment tracking.

