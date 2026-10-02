#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve a pi0-FAST checkpoint on the wire the robot client speaks.
#   $1 checkpoint   $2 dataset root   $3 port   $4 camera names, comma-separated
#
# $2 is REQUIRED: this server reads the normalisation statistics and the checkpoint's
# image-feature list from the dataset. Served with statistics other than the ones it trained on,
# a policy drives confidently to the wrong pose.
#
# Runs in the foreground until killed; the robot client is the `lead` task.
set -euo pipefail

CKPT_IN="${1:?serve.sh needs a checkpoint}"
DS="${2:?serve.sh needs a dataset root}"
PORT="${3:?serve.sh needs a port}"
CAMERAS="${4:?serve.sh needs the camera names}"
HERE="$(cd "$(dirname "$0")" && pwd)"

CKPT="$(pi0fast_resolve_checkpoint "${CKPT_IN}")"
echo "[pi0fast] serve: ${CKPT} on :${PORT} expecting cameras ${CAMERAS}"

# PYTHONPATH so `from transport import PolicyServer` finds transport.py, which the pipeline
# injects beside this script. -u because a traceback from a rejected request would otherwise sit
# in a block-buffered pipe until the lead task tears this down.
PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" exec python -u "${MODEL_FILES}/serve_policy.py" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DS}" \
  --repo-id "${REPO_ID:-local/so101-pi0fast}" \
  --port "${PORT}" \
  --jitter-deg "${JITTER_DEG:-0}"
