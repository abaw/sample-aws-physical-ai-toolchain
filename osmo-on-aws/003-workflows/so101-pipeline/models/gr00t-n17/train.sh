#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fine-tune GR00T N1.7.
#   $1 dataset root   $2 output dir   $3 steps   $4 batch   $5 save-freq   $6 seed
#
# Six arguments because those are the six things a user of ANY model wants to set. Everything
# else here — the base model, the modality config, the embodiment tag, GPU count, dataloader
# workers, whether to use wandb — belongs to GR00T and is therefore in this file, not in the
# pipeline. That split is the whole contract.
set -euo pipefail

DS="${1:?train.sh needs a dataset root}"
OUT="${2:?train.sh needs an output directory}"
STEPS="${3:?train.sh needs a step count}"
BATCH="${4:?train.sh needs a batch size}"
SAVE_FREQ="${5:?train.sh needs a save frequency}"
SEED="${6:?train.sh needs a seed}"

echo "[gr00t-n17] train: steps=${STEPS} batch=${BATCH} save_freq=${SAVE_FREQ} seed=${SEED}"

# THE SEED IS FIXED, NOT ABSENT. GR00T does seed its training: it calls set_seed() and passes
# the value into TrainingArguments. But the value comes from DataConfig.seed, whose default is
# 42, and the curated FinetuneConfig this launcher exposes over tyro has no seed field and does
# not copy one onto the data config. So runs here ARE repeatable to the extent the rest of the
# stack is; they are just repeatable at 42, and this launcher offers no way to ask for another.
#
# Said out loud rather than ignored quietly. The contract lets a script ignore an argument it
# cannot use, but a seed is a promise of reproducibility, and someone passing a seed and getting
# the same run regardless deserves to know which of the two is happening.
export PYTHONHASHSEED="${SEED}"
echo "[gr00t-n17] NOTE: this launcher does not expose a seed. Training is seeded at GR00T's" >&2
echo "[gr00t-n17]       own fixed default (42), so ${SEED} has no effect on the result." >&2

# GR00T's launcher refuses a pre-existing output directory, so the pipeline removes rather
# than creates it; assert that here rather than discovering it inside the launcher.
[ -e "${OUT}" ] && [ -n "$(ls -A "${OUT}" 2>/dev/null)" ] \
  && { echo "ERROR: ${OUT} is not empty; GR00T's launcher refuses that" >&2; exit 1; }

# The wrapper rather than launch_finetune.py: it picks the launcher for one GPU vs many, and
# the single-GPU path avoids a data-parallel wrapper that crashes on this model.
# USE_WANDB=1 needs a WANDB_API_KEY credential and blocks startup on a network call without one.
cd "${GROOT_DIR}"
NUM_GPUS="${NUM_GPUS:-1}" \
MAX_STEPS="${STEPS}" \
SAVE_STEPS="${SAVE_FREQ}" \
GLOBAL_BATCH_SIZE="${BATCH}" \
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}" \
USE_WANDB=0 \
  bash examples/finetune.sh \
    --base-model-path "${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}" \
    --dataset-path "${DS}" \
    --modality-config-path "$(dirname "$0")/common.py" \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir "${OUT}"

echo "[gr00t-n17] train done; checkpoints under ${OUT}"
