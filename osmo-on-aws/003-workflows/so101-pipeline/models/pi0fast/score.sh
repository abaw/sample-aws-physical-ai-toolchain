#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Score a pi0-FAST checkpoint offline and write the pipeline's gate input.
#   $1 checkpoint   $2 dataset root   $3 output dir   $4 traj-ids   $5 horizon   $6 max steps
#
# MUST write $3/metrics.json: the pipeline gates on that file and never reads logs. Here
# score_offline.py writes it directly.
set -euo pipefail

CKPT_IN="${1:?score.sh needs a checkpoint}"
DS="${2:?score.sh needs a dataset root}"
OUT="${3:?score.sh needs an output directory}"
TRAJ_IDS="${4:?score.sh needs trajectory ids}"
HORIZON="${5:?score.sh needs an execution horizon}"
FRAMES="${6:?score.sh needs a per-trajectory frame cap}"

CKPT="$(pi0fast_resolve_checkpoint "${CKPT_IN}")"
echo "[pi0fast] score: ${CKPT} trajs='${TRAJ_IDS}' horizon=${HORIZON} frames_per_trajectory=${FRAMES}"
mkdir -p "${OUT}"

set +e
python -u "${MODEL_FILES}/score_offline.py" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DS}" \
  --repo-id "${REPO_ID:-local/so101-pi0fast}" \
  --traj-ids "${TRAJ_IDS}" \
  --execution-horizon "${HORIZON}" \
  --frames-per-trajectory "${FRAMES}" \
  --warn-reversals "${WARN_REVERSALS:-0.25}" \
  --output-dir "${OUT}" \
  2>&1 | tee "${OUT}/openloop_eval.log"
EVAL_RC=${PIPESTATUS[0]}
set -e
echo "[pi0fast] score exit code: ${EVAL_RC}"
exit "${EVAL_RC}"
