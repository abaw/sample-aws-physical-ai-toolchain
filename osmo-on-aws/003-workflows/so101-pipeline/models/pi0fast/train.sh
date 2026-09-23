#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fine-tune pi0-FAST through lerobot-train.
#   $1 dataset root   $2 output dir   $3 steps   $4 batch   $5 save-freq   $6 seed
#
# Note how little this shares with the other model's train.sh: that one exports environment
# variables and calls a shell launcher after changing directory, this one passes dotted CLI
# flags. The contract is filenames rather than flags precisely because these do not converge.
#
# Unlike that model, this launcher DOES accept a seed, so $6 is honoured rather than reported.
set -euo pipefail

DS="${1:?train.sh needs a dataset root}"
OUT="${2:?train.sh needs an output directory}"
STEPS="${3:?train.sh needs a step count}"
BATCH="${4:?train.sh needs a batch size}"
SAVE_FREQ="${5:?train.sh needs a save frequency}"
SEED="${6:?train.sh needs a seed}"

echo "[pi0fast] train: steps=${STEPS} batch=${BATCH} save_freq=${SAVE_FREQ} seed=${SEED}"

# lerobot-train refuses a pre-existing output directory unless resuming, so it writes to a
# scratch path and the result is moved into place afterwards.
WORK=/tmp/train_out
rm -rf "${WORK}"

# dtype is set explicitly: upstream defaults to float32, which roughly doubles memory on a 2.6B
# model, and inheriting that silently is how a run OOMs for no visible reason. There is no LoRA
# path — pi0_fast is full fine-tune only — so ~15 GB of optimiser state is unavoidable and is
# what caps the batch.
set +e
lerobot-train \
  --dataset.repo_id="${REPO_ID:-local/so101-pi0fast}" \
  --dataset.root="${DS}" \
  --policy.type=pi0_fast \
  --policy.dtype="${DTYPE:-bfloat16}" \
  --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING:-false}" \
  --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS:-30000}" \
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
echo "[pi0fast] lerobot-train exit code: ${TRAIN_RC}"

# Published whether or not training succeeded: a log next to a traceback is what explains an OOM,
# and a task that publishes nothing leaves nothing to diagnose.
if [ -d "${WORK}" ]; then
  cp -a "${WORK}/." "${OUT}/"
fi
exit "${TRAIN_RC}"
