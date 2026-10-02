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
# The stage argument is ignored: every stage needs the same toolchain, because even dataset
# conversion imports gr00t.data for its statistics and modality registry.
#
# Takes ~2 min and produces a ~15 GB .venv, and needs GitHub and the package indexes reachable
# when the task starts. `uv sync` resolves against the uv.lock committed at GROOT_COMMIT.

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

# A checkpoint argument may be one checkpoint directory or a training output holding several.
# Defined here because install.sh is sourced by every stage, so score.sh and serve.sh share it.
#
# `export -f` is required: the stage scripts are EXECUTED, so they inherit exported variables but
# not shell functions. Without it this is `command not found`, two minutes into the install.
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

# ---------------------------------------------------------------------------
# This model's own Python. Add a file here, gated on the stage that uses it; the pipeline
# declares no Python of its own. See docs/CONTRACT.md.
#
# No Jinja delimiters in anything written below, not even in a comment: OSMO renders this file
# as a template, so an opener is consumed before Python ever sees it. submit.sh refuses them.
# ---------------------------------------------------------------------------
MODEL_FILES="${MODEL_FILES:-/tmp/model-files}"
mkdir -p "${MODEL_FILES}"
export MODEL_FILES
# /tmp too: the pipeline injects trajectory_metrics.py and validate_common.py there.
export PYTHONPATH="${MODEL_FILES}:/tmp${PYTHONPATH:+:${PYTHONPATH}}"

# Imported by this model's scripts in every stage.
cat > "${MODEL_FILES}/common.py" <<'MODEL_FILE_EOF'
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
"""GR00T N1.7 modality config for an SO-101 follower arm with two cameras.

Vendored from upstream ``examples/SO100/so100_config.py`` — the SO-101 follower has the
same six joints in the same order — so that this repo owns the contract between training
and the robot. Injected by every GR00T stage, so there is one copy and the stages cannot
disagree.

Registered against ``NEW_EMBODIMENT`` because N1.7's pretraining embodiments are fixed
sets and a new arm is none of them. ``register_modality_config`` asserts the tag is not
already registered, so import this module exactly once per process.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# Joint order is load-bearing: it must match the eval client's servo order, or the
# policy trains to convergence and then drives the wrong joints on real hardware.
#
#   0 shoulder_pan  1 shoulder_lift  2 elbow_flex  3 wrist_flex  4 wrist_roll  5 gripper
#
# The single_arm / gripper split below must equal the one in the dataset's
# meta/modality.json; stage 01 asserts that they agree.
so101_config = {
    # Short keys, mapped to observation.images.* by meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    # Training-time action chunk, and a ceiling on eval: a client's action_horizon must
    # not exceed it, so lowering this caps the closed-loop rate later.
    #
    # It also costs steps. With allow_padding=False the last (chunk - 1) steps of each
    # episode have no full chunk and are dropped, so N episodes yield
    # (total_frames - 15*N) usable steps — expected, not data loss.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            # Arm joints RELATIVE: deltas generalize across absolute poses better than
            # target positions do.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,  # joint space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # Gripper ABSOLUTE: it is effectively binary, and deltas make "stay closed"
            # depend on integrating without drift.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # original_key is task_index, which GR00T resolves through meta/tasks.jsonl.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(so101_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
MODEL_FILE_EOF

case "${STAGE}" in
  prepare)
cat > "${MODEL_FILES}/write_modality.py" <<'MODEL_FILE_EOF'
# SPDX-License-Identifier: Apache-2.0
"""Write meta/modality.json for a LeRobot tree, from the tree's own camera features.

modality.json is a GR00T concept that a stock LeRobot dataset does not carry. It maps the short
video keys GR00T's registered modality config asks for onto the dataset's real feature names.

Both ends are read, never assumed: the short keys come from the registered config, the feature
names come from the dataset's meta/info.json. If a registered key has no matching feature this
refuses rather than guessing -- a modality.json that disagrees with the data is silent at training
time and surfaces as the arm driving on another camera's frames.

Upstream's examples/SO100/modality.json supplies the state, action and annotation blocks, which
are the 5+1 joint layout this arm shares with it.

Argv: dataset root. Env: GROOT_DIR.
"""

import json
import os
import sys
from pathlib import Path

DS = Path(sys.argv[1])
TEMPLATE = Path(os.environ["GROOT_DIR"]) / "examples" / "SO100" / "modality.json"

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: F401  registers NEW_EMBODIMENT on import

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag

out = DS / "meta" / "modality.json"
if out.is_file():
    print(f"[gr00t-n17] {out} already present; leaving it alone")
    raise SystemExit(0)

info = json.loads((DS / "meta" / "info.json").read_text())
features = list(info.get("features", {}))
# LeRobot names a camera feature observation.images.<name>.
cameras = {k.split(".")[-1]: k for k in features if k.startswith("observation.images.")}

wanted = list(MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]["video"].modality_keys)
missing = [k for k in wanted if k not in cameras]
if missing:
    print(
        f"ERROR: this model is configured for cameras {wanted}, but the recording has "
        f"{sorted(cameras) or 'none'}. Re-record with those names, or change the video "
        f"modality_keys in this model's common.py to match the recording.",
        file=sys.stderr,
    )
    raise SystemExit(1)
extra = [k for k in cameras if k not in wanted]
if extra:
    print(f"[gr00t-n17] ignoring camera(s) the model is not configured for: {extra}")

base = json.loads(TEMPLATE.read_text())
# Reuse upstream's entry shape rather than inventing the schema; only the key it points at
# is ours to set.
shape = next(iter(base["video"].values()))
base["video"] = {}
for key in wanted:
    entry = dict(shape)
    for field, value in entry.items():
        if isinstance(value, str) and value.startswith("observation.images."):
            entry[field] = cameras[key]
    base["video"][key] = entry

out.write_text(json.dumps(base, indent=4) + "\n")
print(f"[gr00t-n17] wrote {out} mapping {dict((k, cameras[k]) for k in wanted)}")
MODEL_FILE_EOF

cat > "${MODEL_FILES}/check_dataset.py" <<'MODEL_FILE_EOF'
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
"""GR00T-N1.7-specific dataset checks. Requires the Isaac-GR00T environment.

Runs after validate_common.py, which owns everything model-agnostic. meta/modality.json
is a GR00T concept absent from a stock LeRobot dataset, so it is checked here.

Env: DS (dataset root). Expects common.py alongside this file.
Exits non-zero if any check fails.
"""

import json
import os
import sys
from pathlib import Path

DS = Path(os.environ["DS"])
failures = []


def check(ok, label, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print("== format ==")
# GR00T reads a v2 flavour, not the v3 LeRobot records today. Asserted here rather than
# in validate_common.py because v3 is perfectly valid for a LeRobot-native policy; if
# this fails, the v3 -> v2 downgrade did not run.
info = json.loads((DS / "meta" / "info.json").read_text())
check(
    info["codebase_version"].startswith("v2"),
    "codebase_version is v2.x",
    info["codebase_version"],
)

# modality.json tells GR00T how to slice the state and action arrays. If it disagrees
# with common.py's modality config the mismatch is silent at training time and shows up as the arm
# moving the wrong joint.
print("== GR00T loader ==")
sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402,F401  registers NEW_EMBODIMENT on import

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402

print("== meta/modality.json vs common.py's modality config ==")
check((DS / "meta" / "modality.json").exists(), "meta/modality.json present")
mod = json.loads((DS / "meta" / "modality.json").read_text())
check(
    mod.get("state", {}).get("single_arm") == {"start": 0, "end": 5},
    "state.single_arm is 0-5",
    str(mod.get("state")),
)
check(
    mod.get("state", {}).get("gripper") == {"start": 5, "end": 6},
    "state.gripper is 5-6",
    str(mod.get("state")),
)
check(mod.get("action") == mod.get("state"), "action split matches state split")
# Against the registered config and the dataset's own features, not a hardcoded pair: a
# literal here would pass for any dataset whose modality.json merely contained that literal.
registered = set(MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]["video"].modality_keys)
check(
    set(mod.get("video", {})) == registered,
    f"video keys match the registered config {sorted(registered)}",
    str(sorted(mod.get("video", {}))),
)
dataset_features = set(json.loads((DS / "meta" / "info.json").read_text()).get("features", {}))
mapped = {
    field
    for entry in mod.get("video", {}).values()
    for field in entry.values()
    if isinstance(field, str) and field.startswith("observation.images.")
}
check(
    bool(mapped) and mapped <= dataset_features,
    "every mapped camera feature exists in the dataset",
    f"mapped {sorted(mapped)} against {sorted(f for f in dataset_features if 'images' in f)}",
)
check(
    "human.task_description" in mod.get("annotation", {}),
    "annotation.human.task_description present",
)

print("== GR00T loader ==")
tag = EmbodimentTag.NEW_EMBODIMENT
cfg = MODALITY_CONFIGS[tag.value]
horizon = len(cfg["action"].delta_indices)
print(f"registered modalities {sorted(cfg)}, action horizon {horizon}")

from gr00t.data.stats import generate_rel_stats, generate_stats  # noqa: E402

generate_stats(str(DS))
generate_rel_stats(str(DS), tag)
check((DS / "meta" / "stats.json").exists(), "meta/stats.json generated")
check((DS / "meta" / "relative_stats.json").exists(), "meta/relative_stats.json generated")

from gr00t.data.dataset.sharded_single_step_dataset import (  # noqa: E402
    ShardedSingleStepDataset,
)

ds = ShardedSingleStepDataset(
    dataset_path=str(DS),
    embodiment_tag=tag,
    modality_configs=cfg,
    # 1.0 rather than the 0.1 default: this is a validation pass, so read every step.
    episode_sampling_rate=1.0,
)
check(len(ds) > 0, "dataset shards", f"{len(ds)} shards")
check(ds.action_horizon == horizon, f"dataset action_horizon is {horizon}", str(ds.action_horizon))

# Print the expected step count so the sharder's number can be compared against it. The
# shortfall from total_frames is the action chunk, not data loss — but a genuinely
# unreadable episode also looks like a small shortfall, so it is worth eyeballing.
# Re-read info.json: the downgrade may have rewritten it since the check above.
info = json.loads((DS / "meta" / "info.json").read_text())
expected_steps = info["total_frames"] - (horizon - 1) * info["total_episodes"]
print(
    f"expected usable steps {expected_steps} "
    f"(= {info['total_frames']} - {horizon - 1} x {info['total_episodes']})"
)
# Not checked here: a collated sample. get_shard() needs a model-side processor that only
# DatasetFactory.build() supplies, so construction and sharding are as far as a
# standalone check reaches.

print()
if failures:
    print(f"GR00T VALIDATION FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("GR00T VALIDATION PASSED")
MODEL_FILE_EOF

    ;;
  eval)
cat > "${MODEL_FILES}/measure_smoothness.py" <<'MODEL_FILE_EOF'
#!/usr/bin/env python3
"""Measure how smooth a checkpoint's predicted action chunks are, against the ground truth.

**Why this exists as a separate number from MSE.** Error against ground truth does not see
jitter: a zero-mean oscillation superimposed on a broadly correct trajectory scores well on
MSE and MAE, and a checkpoint that scored MAE 1.55 here produced chunks that reversed
direction on half their steps. On a real arm that is not a cosmetic defect — the arm does
the same amount of servo work while oscillating instead of progressing, and the smallest
joint reached 60 C in two minutes of it, which is close to the temperature at which these
servos cut their own torque and drop the arm.

So this reports the fraction of consecutive steps on which each joint reverses direction,
next to the same figure for the trajectory being imitated. Teleop recordings sit under 1%;
they also hold several joints *exactly* still more than half the time. A policy far above
that is usable, but only with a command deadband to suppress the dither — see stage 04.

Reuses open_loop_eval.py's dataset and observation path, so these are the chunks stage 03
scores. GPU only; touches no hardware. Pass a comma-separated list of denoising-step counts
to sweep them; the default measures the one value given.
"""

from copy import deepcopy
import json
import pathlib
import sys
import time

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.eval.open_loop_eval import parse_action_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Shared with every other model's scorer, so a statistic two pipelines report under the same
# name is the same code.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from trajectory_metrics import summarize  # noqa: E402


def main() -> int:
    dataset_path = sys.argv[1]
    model_path = sys.argv[2]
    traj_id = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 160
    sweep = [int(v) for v in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["4", "8", "16", "32"])]

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve("new_embodiment"),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    horizon = len(policy.modality_configs["action"].delta_indices)
    print(f"chunk length {horizon}, device {torch.cuda.get_device_name(0)}", flush=True)

    loader = LeRobotEpisodeLoader(dataset_path=dataset_path, modality_configs=policy.modality_configs)
    traj = loader[traj_id]
    actual = min(steps, len(traj))
    obs_configs = deepcopy(policy.modality_configs)
    obs_configs.pop("action")
    action_keys = policy.modality_configs["action"].modality_keys

    # Ground truth over the same window, as the reference every row is judged against.
    truth = []
    for step in range(actual):
        point = extract_step_data(traj, step, policy.modality_configs, EmbodimentTag.NEW_EMBODIMENT)
        truth.append(
            np.concatenate([np.atleast_1d(point.actions[k][0]) for k in action_keys], axis=0)
        )
    truth = np.stack(truth)  # (actual, 6)
    truth_smooth = summarize([truth], JOINTS)
    print("\nground truth (the trajectory the policy is imitating):")
    for name, v in truth_smooth.items():
        print(f"  {name:15s} reversals {v['reversals']:6.1%}   |step| p50 {v['step_p50']:.3f}", flush=True)

    results = {"ground_truth": truth_smooth, "chunk_length": horizon, "sweep": {}}

    for denoise in sweep:
        policy.model.action_head.num_inference_timesteps = denoise
        chunks, latencies, errors = [], [], []

        # Warm up at this setting; the first sample after changing it pays kernel cost.
        first = extract_step_data(traj, 0, obs_configs, EmbodimentTag.NEW_EMBODIMENT)
        warm = {}
        for k, v in first.states.items():
            warm[f"state.{k}"] = v
        for k, v in first.images.items():
            warm[f"video.{k}"] = np.array(v)
        for key in policy.modality_configs["language"].modality_keys:
            warm[key] = first.text
        for _ in range(3):
            policy.get_action(parse_observation_gr00t(warm, policy.modality_configs))

        for step in range(0, actual - horizon, horizon):
            point = extract_step_data(traj, step, obs_configs, EmbodimentTag.NEW_EMBODIMENT)
            obs = {}
            for k, v in point.states.items():
                obs[f"state.{k}"] = v
            for k, v in point.images.items():
                obs[f"video.{k}"] = np.array(v)
            for key in policy.modality_configs["language"].modality_keys:
                obs[key] = point.text

            start = time.perf_counter()
            raw, _ = policy.get_action(parse_observation_gr00t(obs, policy.modality_configs))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

            parsed = parse_action_gr00t(raw)
            chunk = np.stack(
                [
                    np.concatenate(
                        [np.atleast_1d(np.atleast_1d(parsed[f"action.{k}"])[t]) for k in action_keys],
                        axis=0,
                    )
                    for t in range(horizon)
                ]
            )  # (T, 6)
            chunks.append(chunk)
            errors.append(chunk - truth[step : step + horizon])

        smooth = summarize(chunks, JOINTS)
        err = np.concatenate(errors, axis=0)
        row = {
            "denoising_steps": denoise,
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1),
            "mse": round(float((err**2).mean()), 3),
            "mae": round(float(np.abs(err).mean()), 3),
            "joints": smooth,
            "worst_reversals": round(max(v["reversals"] for v in smooth.values()), 4),
        }
        results["sweep"][str(denoise)] = row
        print(
            f"\ndenoising {denoise:3d}: latency p50 {row['latency_p50_ms']:6.1f} ms  "
            f"MSE {row['mse']:8.3f}  MAE {row['mae']:6.3f}  worst reversals "
            f"{row['worst_reversals']:.1%}",
            flush=True,
        )
        for name, v in smooth.items():
            print(
                f"  {name:15s} reversals {v['reversals']:6.1%}   |step| p50 {v['step_p50']:.3f}",
                flush=True,
            )

    print("\nRESULTS_JSON " + json.dumps(results), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
MODEL_FILE_EOF

    ;;
  serve)
cat > "${MODEL_FILES}/serve_policy.py" <<'MODEL_FILE_EOF'
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
"""Serve a fine-tuned GR00T checkpoint over ZMQ for the closed-loop eval client.

Thin wrapper over upstream's ``gr00t/eval/run_gr00t_server.py``, adding the one thing
stage 04 cannot do without: **an explicit denoising-step count**. Upstream's own
open-loop evaluator refuses ``--denoising-steps`` against a remote server and says to
"set the denoising steps on the server instead", but the server script exposes no such
flag, so the value silently falls back to the checkpoint's config. Denoising steps are
the largest single lever on inference latency, and therefore on whether the control
loop closes at all, so stage 04 must serve at the same value stage 03 evaluated at.

Also logs the served action-chunk length. That number is the ceiling on the client's
``--action-horizon``, and having it in the server log means a horizon rejection can be
diagnosed from either side.

The checkpoint must be a fine-tuned one. The base model has no ``new_embodiment``
entry in its processor and raises at construction — that tag exists only for
fine-tuning.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Fine-tuned checkpoint directory")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - pod-local, one port
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=4,
        help="Diffusion sampling steps. The action head reads this at sampling time.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Skip server-side observation/action validation. Leave it on: it is what "
        "turns a malformed observation into a named error instead of a wrong action.",
    )
    parser.add_argument(
        "--expect-cameras",
        default=None,
        help="Comma-separated camera names the robot will send. Compared against the "
        "checkpoint's own video keys, and a mismatch refuses to serve. Without this the "
        "server accepts an extra camera and silently drops it, which is a wrong-input "
        "bug no metric downstream can see.",
    )
    args = parser.parse_args()

    import torch

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    if not torch.cuda.is_available():
        # A CPU fallback would load and then serve at several seconds per chunk, which
        # looks like a network problem from the client's side.
        print("ERROR: no CUDA device visible; the policy server needs a GPU", file=sys.stderr)
        return 1

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}", flush=True)

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        model_path=args.model_path,
        device="cuda",
        strict=not args.no_strict,
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps

    action_config = policy.modality_configs["action"]
    chunk_length = len(action_config.delta_indices)
    print(f"denoising steps: {args.denoising_steps}", flush=True)
    print(f"action chunk length: {chunk_length} (client --action-horizon must not exceed it)", flush=True)
    print(f"action keys: {action_config.modality_keys}", flush=True)
    print(f"state keys: {policy.modality_configs['state'].modality_keys}", flush=True)
    video_keys = list(policy.modality_configs["video"].modality_keys)
    print(f"video keys: {video_keys}", flush=True)

    # The server holds the checkpoint, so it is the only component that can tell whether
    # the rig about to be pointed at it is the one the policy was trained on. Upstream does
    # not check: an extra camera is dropped and a run completes on an input the policy never
    # saw, which no metric downstream can detect. Refusing here happens before the robot
    # task enables torque.
    if args.expect_cameras:
        expected = [c for c in args.expect_cameras.split(",") if c]
        if set(expected) != set(video_keys):
            print(
                f"ERROR: this checkpoint was trained on cameras {sorted(video_keys)}, but the "
                f"robot is configured to send {sorted(expected)} — refusing to serve. Extra "
                f"cameras are silently dropped and a missing one cannot be substituted, so "
                f"either would drive the arm from an input the policy was not trained on.",
                file=sys.stderr,
                flush=True,
            )
            return 1
        print(f"camera set matches the checkpoint: {sorted(expected)}", flush=True)

    # Blocks forever. The client carries `lead: true`, so the group ends when the
    # client exits and this task is torn down with it.
    PolicyServer.start_server(policy=policy, port=args.port, host=args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
MODEL_FILE_EOF

    ;;
esac

echo "[gr00t-n17] model files in ${MODEL_FILES}: $(ls "${MODEL_FILES}" | tr '\n' ' ')"
