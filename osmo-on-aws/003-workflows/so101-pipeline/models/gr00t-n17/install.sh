#!/usr/bin/env bash
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# Install Isaac-GR00T N1.7 into a generic NGC PyTorch container.  $1 = stage name.
#
# SOURCED, not executed — it exports VIRTUAL_ENV / PATH / PYTHONPATH / GROOT_DIR:
#
#     source /tmp/install.sh prepare
#
# **The stage argument is accepted and ignored**, which the contract allows. GR00T's dataset
# conversion imports gr00t.data for statistics and the modality registry, so `prepare` needs the
# same toolchain the training stage does; gating it would save nothing and would leave two code
# paths where one is exercised. A model whose prepare is pure parquet work should gate on $1.
#
# Takes ~2 min and produces a ~15 GB .venv. Installing at runtime rather than baking GR00T into an
# image is deliberate: `uv sync` resolves against the uv.lock committed at GROOT_COMMIT, so it is
# reproducible rather than merely convenient, and an image carrying the venv is roughly twice the
# download of the base alone. The cost is those minutes per task, and a dependency on GitHub and
# the package indexes being reachable when the task starts.

set -euo pipefail

STAGE="${1:?install.sh needs the stage name}"
echo "[gr00t-n17] install for stage: ${STAGE}"

# Isaac-GR00T N1.7 General Release. Pinned here, which is the only place it is pinned.
GROOT_COMMIT="${GROOT_COMMIT:-1a1837f20538b7d7e21f977a11a5aee14f99803c}"
GROOT_DIR="${GROOT_DIR:-/tmp/gr00t-repo}"

# The model's own system dependencies, installed by the model. The pipeline does not know that
# GR00T needs ffmpeg — torchcodec raises "Could not load libtorchcodec" without it, and git-lfs
# is needed because uv resolves a wheel tracked in LFS.
if ! command -v git-lfs >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git git-lfs ffmpeg
fi

if [ ! -d "${GROOT_DIR}/.git" ]; then
  echo "[gr00t-n17] cloning Isaac-GR00T at ${GROOT_COMMIT}"
  git clone --quiet https://github.com/NVIDIA/Isaac-GR00T.git "${GROOT_DIR}"
  git -C "${GROOT_DIR}" checkout --quiet "${GROOT_COMMIT}"
  # A pointer file left in place of the real wheel fails `uv sync` with
  # "Invalid zip file structure", which says nothing about LFS.
  git -C "${GROOT_DIR}" lfs pull
fi
git -C "${GROOT_DIR}" log -1 --format='[gr00t-n17] %H %ai %s'

echo "[gr00t-n17] uv sync"
pip install --quiet uv
( cd "${GROOT_DIR}" && uv sync )

export GROOT_DIR
export VIRTUAL_ENV="${GROOT_DIR}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${GROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Fail here rather than deep in a training run.
python - <<'PY'
import torch, gr00t, torchcodec, flash_attn
print(f"[gr00t-n17] torch {torch.__version__} cuda {torch.version.cuda} "
      f"devices {torch.cuda.device_count()}")
print(f"[gr00t-n17] torchcodec {torchcodec.__version__} flash_attn {flash_attn.__version__}")
print(f"[gr00t-n17] gr00t {gr00t.__file__}")
PY

echo "[gr00t-n17] toolchain ready for stage: ${STAGE}"

# A checkpoint argument may be a single checkpoint directory or the whole training output that
# holds several. Resolving that is GR00T's business, not the pipeline's, so it lives here as a
# function — install.sh is sourced by every stage, which makes this the one place both the
# scoring and serving scripts can share it without a second copy.
#
# `export -f` is not decoration. The stage scripts are EXECUTED, not sourced, so they are child
# processes and inherit exported variables but not shell functions; without the export this
# resolves to `command not found` only at run time, after a two-minute toolchain install.
#
# -V because a lexical sort puts checkpoint-900 after checkpoint-1000.
gr00t_resolve_checkpoint() {
  local given="${1:?gr00t_resolve_checkpoint needs a path}"
  if [ -f "${given}/config.json" ]; then
    echo "${given}"
    return 0
  fi
  local found
  found="$(find "${given}" -type d -name 'checkpoint-*' | sort -V | tail -1)"
  if [ -z "${found}" ]; then
    echo "ERROR: no checkpoint-<step> directory under ${given}" >&2
    find "${given}" -maxdepth 3 -type d >&2
    return 1
  fi
  echo "${found}"
}
export -f gr00t_resolve_checkpoint
