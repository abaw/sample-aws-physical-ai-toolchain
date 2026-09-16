#!/usr/bin/env bash
# LeRobot v0.6.1 into a venv, for pi0-FAST training. Sourced, not executed: exports
# LEROBOT_DIR and puts the venv on PATH, mirroring so101-vla/files/gr00t/install_n17.sh.
#
# Pinned to a tag, not main. v0.5.1 -> v0.6.1 landed between April and August 2026 and
# removed GR00T N1.5 support outright, so this tree moves fast enough to break a pipeline
# between two runs of the same workflow.
#
# Extras, and why these and not `lerobot[pi]`: modeling_pi0_fast.py calls
# require_package("transformers", extra="pi") and require_package("scipy", extra="pi"),
# but **there is no `pi` extra in pyproject.toml at v0.6.1** — the error message names a
# group that does not exist. So the real dependencies are named directly. `dataset` is for
# the v2.1 -> v3.0 conversion script, which needs jsonlines.
set -euo pipefail

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
uv pip install -e ".[transformers-dep,scipy-dep,dataset,training]"

python - <<'PY'
import importlib, torch
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
c = PI0FastConfig()
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
print(f"transformers {importlib.import_module('transformers').__version__}")
# Printed rather than asserted: if upstream changes a default, the run should say so in the
# log rather than fail a check nobody remembers writing.
print(
    f"pi0_fast defaults: dtype={c.dtype} chunk={c.chunk_size} n_action_steps={c.n_action_steps} "
    f"max_action_dim={c.max_action_dim} grad_ckpt={c.gradient_checkpointing} "
    f"lr={c.optimizer_lr} warmup={c.scheduler_warmup_steps}"
)
PY

export LEROBOT_DIR
echo "LEROBOT_DIR=${LEROBOT_DIR} (venv on PATH)"
