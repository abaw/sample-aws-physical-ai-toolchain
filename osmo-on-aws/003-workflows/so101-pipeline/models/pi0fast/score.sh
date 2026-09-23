#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Score a pi0-FAST checkpoint offline and write the pipeline's gate input.
#   $1 checkpoint   $2 dataset root   $3 output dir   $4 traj-ids   $5 horizon   $6 max steps
#
# score_lib.py writes $3/metrics.json itself rather than printing numbers a caller has to scrape,
# so the artifact cannot disagree with the log. The other shipped model has to scrape, because
# its upstream evaluator only prints — which is exactly why the contract asks for the FILE and
# leaves the method to the model.
set -euo pipefail

CKPT_IN="${1:?score.sh needs a checkpoint}"
DS="${2:?score.sh needs a dataset root}"
OUT="${3:?score.sh needs an output directory}"
TRAJ_IDS="${4:?score.sh needs trajectory ids}"
HORIZON="${5:?score.sh needs an execution horizon}"
MAX_STEPS="${6:?score.sh needs a step cap}"
HERE="$(cd "$(dirname "$0")" && pwd)"

CKPT="$(pi0fast_resolve_checkpoint "${CKPT_IN}")"
echo "[pi0fast] score: ${CKPT} trajs='${TRAJ_IDS}' horizon=${HORIZON} max_steps=${MAX_STEPS}"
mkdir -p "${OUT}"

set +e
python -u "${HERE}/score_lib.py" \
  --checkpoint "${CKPT}" \
  --dataset-root "${DS}" \
  --repo-id "${REPO_ID:-local/so101-pi0fast}" \
  --traj-ids "${TRAJ_IDS}" \
  --execution-horizon "${HORIZON}" \
  --max-steps "${MAX_STEPS}" \
  --warn-reversals "${WARN_REVERSALS:-0.25}" \
  --output-dir "${OUT}" \
  2>&1 | tee "${OUT}/openloop_eval.log"
EVAL_RC=${PIPESTATUS[0]}
set -e
echo "[pi0fast] score exit code: ${EVAL_RC}"
exit "${EVAL_RC}"
