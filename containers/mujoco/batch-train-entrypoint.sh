#!/bin/bash
# =============================================================================
# MuJoCo Playground AWS Batch Training Entrypoint
#
# Single-node, single-GPU Brax PPO training on a MuJoCo Playground (MJX) env.
# The job definition runs this as `batch-train`; train_entrypoint.py does the
# training, writes checkpoints and a rollout MP4, and uploads them to S3.
#
# Environment variables (set by job definition or container overrides):
#   ENV_NAME          - Playground env name (default: CartpoleBalance)
#   NUM_ENVS          - Parallel environments (default: 2048)
#   NUM_TIMESTEPS     - Total environment steps (default: 5000000)
#   SEED              - Random seed (default: 0)
#   CHECKPOINT_BUCKET - S3 bucket for checkpoints and the MP4 (default: none)
#   CHECKPOINT_PREFIX - S3 key prefix (default: checkpoints/<job_id>)
# =============================================================================

set -e

# Headless rendering for the rollout MP4 (the image sets this too)
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec python3 "$(dirname "$0")/train_entrypoint.py"
