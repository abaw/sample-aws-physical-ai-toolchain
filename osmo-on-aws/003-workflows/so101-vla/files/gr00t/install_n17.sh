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

# Installs Isaac-GR00T N1.7 into a generic NGC PyTorch container.
#
# Source it, don't execute it — it exports VIRTUAL_ENV / PATH / PYTHONPATH:
#
#     source /tmp/install_n17.sh
#
# Every so101-vla stage injects this file, so GROOT_COMMIT is pinned in one place.
# Changing the pin here changes it everywhere.
#
# Takes ~2 minutes and produces a ~15 GB .venv. Installing at runtime rather than baking
# GR00T into an image is deliberate: `uv sync` resolves against the uv.lock committed at
# GROOT_COMMIT, so it is reproducible rather than merely convenient, and an image
# carrying the venv is roughly twice the download of the base alone. The cost is these
# ~2 minutes per task, and a dependency on GitHub and the package indexes being
# reachable when the task starts.
#
# Does not apt-install anything: git, git-lfs and ffmpeg are also needed by the
# model-agnostic half of stage 01, so the workflow installs them once before branching.

set -euo pipefail

# Isaac-GR00T N1.7 General Release.
GROOT_COMMIT="${GROOT_COMMIT:-1a1837f20538b7d7e21f977a11a5aee14f99803c}"
GROOT_DIR="${GROOT_DIR:-/tmp/gr00t-repo}"

if [ ! -d "${GROOT_DIR}/.git" ]; then
  echo "[install_n17] cloning Isaac-GR00T at ${GROOT_COMMIT}"
  git clone --quiet https://github.com/NVIDIA/Isaac-GR00T.git "${GROOT_DIR}"
  git -C "${GROOT_DIR}" checkout --quiet "${GROOT_COMMIT}"
fi
git -C "${GROOT_DIR}" log -1 --format='[install_n17] %H %ai %s'

echo "[install_n17] uv sync"
pip install --quiet uv
( cd "${GROOT_DIR}" && uv sync )

export GROOT_DIR
export VIRTUAL_ENV="${GROOT_DIR}/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${GROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Fail here rather than deep in a training run. torchcodec in particular raises
# "Could not load libtorchcodec" when ffmpeg is missing or out of its 4-7 range.
python - <<'PY'
import torch, gr00t, torchcodec, flash_attn
print(f"[install_n17] torch {torch.__version__} cuda {torch.version.cuda} "
      f"devices {torch.cuda.device_count()}")
print(f"[install_n17] torchcodec {torchcodec.__version__} flash_attn {flash_attn.__version__}")
print(f"[install_n17] gr00t {gr00t.__file__}")
PY

echo "[install_n17] ready"
