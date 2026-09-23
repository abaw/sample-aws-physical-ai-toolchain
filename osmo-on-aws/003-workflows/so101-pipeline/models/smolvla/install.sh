#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install LeRobot with the SmolVLA policy.  $1 = stage name.  SOURCED, not executed.
#
# The stage argument is accepted and ignored: every stage here needs the same LeRobot venv,
# because even dataset conversion runs LeRobot`s own converter script. A model whose prepare
# is pure parquet work should gate on $1 and install less.
set -euo pipefail
STAGE="${1:?install.sh needs the stage name}"
echo "[smolvla] install for stage: ${STAGE}"

# The model's own system dependencies, installed by the model. The pipeline does not know that
# this toolchain needs curl to fetch itself, ffmpeg for torchcodec (without libav* its library
# fails to load and LeRobot silently falls back to another decoder, so training and serving would
# see frames decoded differently), or uv to resolve the venv.
if ! command -v curl >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq curl ffmpeg
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

LEROBOT_TAG="${LEROBOT_TAG:-v0.6.1}"
LEROBOT_DIR="${LEROBOT_DIR:-/tmp/lerobot}"

# No git in the base image on the GR00T path, and a tarball is faster than a clone anyway.
mkdir -p "${LEROBOT_DIR}"
curl -fsSL "https://github.com/huggingface/lerobot/archive/refs/tags/${LEROBOT_TAG}.tar.gz" \
  | tar -xz --strip-components=1 -C "${LEROBOT_DIR}"

cd "${LEROBOT_DIR}"

# One resolve, not several. Separate pip installs let a later one silently downgrade an
# earlier one's pins — that cost a run on the GR00T side when huggingface-hub moved.
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[smolvla,scipy-dep,dataset,training]"

python - <<'PY'
import importlib, torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
c = SmolVLAConfig()
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
print(f"transformers {importlib.import_module('transformers').__version__}")
# Printed rather than asserted: if upstream changes a default the run should say so in the log
# rather than fail a check nobody remembers writing. Note there is no dtype field here, unlike
# the other LeRobot policy in this repo, and freeze_vision_encoder defaults True, which is most
# of why this model trains so much faster.
print(
    f"smolvla defaults: chunk={c.chunk_size} n_action_steps={c.n_action_steps} "
    f"max_action_dim={c.max_action_dim} vlm={c.vlm_model_name} "
    f"freeze_vision_encoder={c.freeze_vision_encoder} load_vlm_weights={c.load_vlm_weights} "
    f"lr={c.optimizer_lr} warmup={c.scheduler_warmup_steps}"
)
PY

export LEROBOT_DIR
echo "LEROBOT_DIR=${LEROBOT_DIR} (venv on PATH)"
# The serving stage additionally needs the transport's own dependencies inside this venv. This
# is what the stage argument is FOR: the other stages never open a socket, so they do not pay for
# it, and the transport slot itself declares no dependencies because a file cannot.
if [ "${STAGE}" = "serve" ]; then
  uv pip install "pyzmq==${PYZMQ_VERSION:-27.0.1}" "msgpack==${MSGPACK_VERSION:-1.1.0}"
  echo "[smolvla] transport dependencies installed for serving"
fi

echo "[smolvla] toolchain ready for stage: ${STAGE}"

# A checkpoint argument may be a pretrained_model/ directory or a training output holding several
# steps. Resolving that is this model's business, not the pipeline's. Exported, not merely
# defined: the stage scripts are EXECUTED, so they are child processes and inherit exported
# variables and functions but not plain ones.
smolvla_resolve_checkpoint() {
  local given="${1:?smolvla_resolve_checkpoint needs a path}"
  if [ -f "${given}/config.json" ]; then
    echo "${given}"
    return 0
  fi
  local found
  # -V because a lexical sort puts 090000 before 100000 only by luck of zero padding.
  found="$(find "${given}" -type d -name pretrained_model | sort -V | tail -1)"
  if [ -z "${found}" ]; then
    echo "ERROR: no pretrained_model/ directory under ${given}" >&2
    find "${given}" -maxdepth 4 -type d >&2
    return 1
  fi
  echo "${found}"
}
export -f smolvla_resolve_checkpoint
