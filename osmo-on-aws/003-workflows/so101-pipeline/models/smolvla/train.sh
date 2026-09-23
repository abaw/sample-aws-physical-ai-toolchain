#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fine-tune SmolVLA through lerobot-train.
#   $1 dataset root   $2 output dir   $3 steps   $4 batch   $5 save-freq   $6 seed
#
# The one place this model genuinely diverges from the other LeRobot-native one here:
#
#   --policy.path=lerobot/smolvla_base   rather than   --policy.type=smolvla
#
# `--policy.path` loads the pretrained SmolVLA weights AND its config, which is what "fine-tune"
# means for this model; `--policy.type` would build an action expert from scratch against a frozen
# VLM, and the config field that distinguishes them (load_vlm_weights) defaults to False. Passing
# the type instead of the path trains a different thing and reports success either way.
#
# There is also no --policy.dtype: SmolVLAConfig has no dtype field, unlike the other model here,
# whose default of float32 had to be overridden to avoid doubling memory. Passing it would be
# rejected by the CLI.
set -euo pipefail

DS="${1:?train.sh needs a dataset root}"
OUT="${2:?train.sh needs an output directory}"
STEPS="${3:?train.sh needs a step count}"
BATCH="${4:?train.sh needs a batch size}"
SAVE_FREQ="${5:?train.sh needs a save frequency}"
SEED="${6:?train.sh needs a seed}"

echo "[smolvla] train: steps=${STEPS} batch=${BATCH} save_freq=${SAVE_FREQ} seed=${SEED}"

WORK=/tmp/train_out
rm -rf "${WORK}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# The pretrained checkpoint expects its image features named camera1..N; a recording names them
# whatever it named them. Derived from the DATASET, which is the authority on what cameras exist —
# never from a workflow variable, which could disagree with the recording. The rule lives in
# common.py so serving applies the identical mapping; see the note there.
RENAME_MAP="$(PYTHONPATH="${HERE}" python - "${DS}/meta/info.json" <<'PY'
import json, sys
from common import image_keys_from_info, rename_map
info = json.load(open(sys.argv[1]))
print(json.dumps(rename_map(image_keys_from_info(info))))
PY
)"
echo "[smolvla] rename_map: ${RENAME_MAP}"

# ~450M parameters against the other model's 2.6B, and the vision encoder is frozen by default,
# so this fits a far larger batch on the same card. The default here is still the pipeline's, which
# a user raises for this model rather than the pipeline guessing per model.
set +e
lerobot-train \
  --dataset.repo_id="${REPO_ID:-local/so101-smolvla}" \
  --dataset.root="${DS}" \
  --policy.path="${BASE_POLICY_PATH:-lerobot/smolvla_base}" \
  --rename_map="${RENAME_MAP}" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size="${BATCH}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ:-100}" \
  --seed="${SEED}" \
  --output_dir="${WORK}" \
  2>&1 | tr '\r' '\n' | tee "${OUT}/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
echo "[smolvla] lerobot-train exit code: ${TRAIN_RC}"

# Published whether or not training succeeded: a log beside a traceback is what explains an OOM.
if [ -d "${WORK}" ]; then
  cp -a "${WORK}/." "${OUT}/"
fi
exit "${TRAIN_RC}"
