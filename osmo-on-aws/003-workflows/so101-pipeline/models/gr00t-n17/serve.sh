#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Serve a GR00T N1.7 checkpoint on the wire the robot client speaks.
#   $1 checkpoint   $2 dataset root   $3 port   $4 camera names, comma-separated
#
# $2 is ACCEPTED AND IGNORED: this checkpoint carries its own normalisation statistics, so the
# server needs no dataset. Every script receives every argument whether it uses it or not.
#
# Runs in the foreground until killed. The robot client is the `lead` task, so its exit tears
# this down.
set -euo pipefail

CKPT_IN="${1:?serve.sh needs a checkpoint}"
DS="${2-}"          # unused; see above
PORT="${3:?serve.sh needs a port}"
CAMERAS="${4:?serve.sh needs the camera names}"

CKPT="$(gr00t_resolve_checkpoint "${CKPT_IN}")"
echo "[gr00t-n17] serve: ${CKPT} on :${PORT} expecting cameras ${CAMERAS}"

if [ ! -f "${CKPT}/experiment_cfg/dataset_statistics.json" ]; then
  echo "ERROR: ${CKPT} has no experiment_cfg/dataset_statistics.json — a checkpoint without normalisation statistics serves confidently at the wrong scale" >&2
  exit 1
fi

cd "${GROOT_DIR}"
# -u because the server's own error handler prints a traceback and then keeps serving, so its
# stdout is a pipe that stays open: block buffering means the traceback for a rejected request
# never reaches the task log.
exec python -u "${MODEL_FILES}/serve_policy.py" \
  --model-path "${CKPT}" \
  --embodiment-tag NEW_EMBODIMENT \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --expect-cameras "${CAMERAS}" \
  --denoising-steps "${DENOISING_STEPS:-4}"
