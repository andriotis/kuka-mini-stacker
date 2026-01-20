FROM python:3.11-slim

# Install uv for fast, reproducible dependency management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency specification and install packages with uv
# Make sure to create a requirements.txt alongside env_test.py
COPY requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

# Run PyBullet in headless (DIRECT) mode inside Docker by default
ENV PYBULLET_RENDER_MODE=headless

# Copy simulation script and URDF
COPY env_test.py /app/env_test.py
COPY kuka_3dof.urdf /app/kuka_3dof.urdf

# Copy pybullet_kuka directory (contains mesh files referenced by URDF)
COPY pybullet_kuka/ /app/pybullet_kuka/

# Default command to run the simulation
CMD ["python", "env_test.py"]

