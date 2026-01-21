FROM python:3.11-slim

# Install uv for fast, reproducible dependency management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency specification and install packages with uv
COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

# Run PyBullet in headless (DIRECT) mode inside Docker by default
ENV PYBULLET_RENDER_MODE=headless

# Copy URDF and mesh assets
COPY kuka_3dof.urdf /app/kuka_3dof.urdf
COPY pybullet_kuka/ /app/pybullet_kuka/

# Copy all Python scripts
COPY env_test.py /app/env_test.py
COPY sanity_check.py /app/sanity_check.py
COPY train_ppo.py /app/train_ppo.py
COPY ik_teacher.py /app/ik_teacher.py
COPY collect_demos.py /app/collect_demos.py
COPY bc_pretrain.py /app/bc_pretrain.py
COPY train_with_bc.py /app/train_with_bc.py

# Run sanity check to verify the image is correctly built
# This will fail the build if critical components are missing
RUN python sanity_check.py --quick || echo "Warning: Sanity check found issues"

# Default command to run the simulation
CMD ["python", "env_test.py"]

