#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve a pi0-FAST checkpoint on the wire the robot client speaks.
#   $1 checkpoint   $2 dataset root   $3 port   $4 camera names, comma-separated
#
# $2 is REQUIRED here, unlike the other shipped model: this server reads normalisation statistics
# and the checkpoint's image-feature list from the dataset, and a policy served with different
# statistics than it trained on drives confidently to the wrong pose. That asymmetry is why the
# contract passes the dataset to every model's serve.sh whether or not it is used.
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

# PYTHONPATH so `from transport import PolicyServer` resolves to the transport slot beside this
# script. -u because a traceback from a rejected request would otherwise sit in a block-buffered
# pipe until the lead task tears this down.
PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" exec python -u "${HERE}/serve_lib.py" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DS}" \
  --repo-id "${REPO_ID:-local/so101-pi0fast}" \
  --port "${PORT}" \
  --jitter-deg "${JITTER_DEG:-0}"
