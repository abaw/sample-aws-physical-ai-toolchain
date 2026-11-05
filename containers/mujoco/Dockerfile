# MuJoCo Training Container
#
# MuJoCo + MJX (the JAX port) + MuJoCo Playground (Brax PPO) on a public CUDA
# base. Everything installs from PyPI, so the build needs no NGC key.
#
# Build (from the repo root, which is the build context):
#   docker build -t mujoco-training:latest -f containers/mujoco/Dockerfile .
#
# Run (smoke check):
#   docker run --gpus all --rm mujoco-training:latest \
#     python -c "import mujoco, mujoco.mjx, mujoco_playground, jax; print(mujoco.__version__, jax.__version__)"

# Base: public CUDA base (driver libs come from the host via the NVIDIA
# container toolkit; jax[cuda12] brings its own CUDA/cuDNN wheels).
# Ubuntu 24.04 ships Python 3.12.
FROM nvidia/cuda:12.9.2-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python, plus the GL/EGL libraries MuJoCo's offscreen renderer loads and
# ffmpeg for writing rollout MP4s
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    libegl1 \
    libgl1 \
    libopengl0 \
    libgles2 \
    libglvnd0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04's system Python is externally managed (PEP 668), so install into a venv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="/opt/venv/bin:${PATH}"

# Install MuJoCo, MJX, Playground and JAX, pinned to exact versions.
# brax 0.14.2 (Playground's PPO) calls jax.device_put_replicated, which
# jax 0.10 turned into an error, so jax stays on 0.9.x until brax moves.
RUN pip install --no-cache-dir \
    mujoco==3.17.0 \
    mujoco-mjx==3.17.0 \
    playground==0.2.0 \
    brax==0.14.2 \
    "jax[cuda12]==0.9.2" \
    boto3==1.43.124

# Headless rendering through EGL. The NVIDIA container toolkit mounts
# libEGL_nvidia.so.0 when the graphics capability is requested; this vendor
# file is the fallback glvnd uses when the runtime injects none of its own.
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Copy training entrypoints
COPY containers/mujoco/batch-train-entrypoint.sh /opt/ml/code/batch-train
COPY containers/mujoco/train_entrypoint.py /opt/ml/code/train_entrypoint.py
RUN chmod +x /opt/ml/code/batch-train
ENV PATH="/opt/ml/code:${PATH}"

# Copy training code
COPY training/ /workspace/training/

WORKDIR /workspace

ENTRYPOINT []
CMD ["python"]
