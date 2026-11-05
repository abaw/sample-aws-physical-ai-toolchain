"""
CPU smoke run of the MuJoCo training entrypoint.

Runs batch-train-entrypoint.sh with a tiny config on the CPU and no bucket, and
checks it leaves a checkpoint directory and a rollout MP4. Needs the image's
Python packages (mujoco, mujoco-mjx, playground, jax), an offscreen GL
(MUJOCO_GL=egl or osmesa) and ffmpeg or imageio-ffmpeg:
  pytest containers/mujoco/test_train_entrypoint.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco_playground")

ENTRYPOINT = Path(__file__).parent / "batch-train-entrypoint.sh"


def test_cpu_smoke_run_writes_checkpoint_and_mp4(tmp_path):
    env = os.environ.copy()
    env.pop("CHECKPOINT_BUCKET", None)
    env.update(
        JAX_PLATFORMS="cpu",
        NUM_ENVS="8",
        NUM_TIMESTEPS="8000",
        SEED="0",
        OUTPUT_DIR=str(tmp_path),
        # The entrypoint runs python3; use this interpreter's environment
        PATH=f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}",
    )

    result = subprocess.run([str(ENTRYPOINT)], env=env, timeout=1800)

    assert result.returncode == 0
    checkpoints = [p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir()]
    assert checkpoints, "no checkpoint directory written"
    assert any(any(c.iterdir()) for c in checkpoints), "checkpoint directory is empty"
    mp4 = tmp_path / "rollout.mp4"
    assert mp4.is_file() and mp4.stat().st_size > 0
