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
check(
    set(mod.get("video", {})) == {"front", "wrist"},
    "video keys are front + wrist",
    str(list(mod.get("video", {}))),
)
check(
    "human.task_description" in mod.get("annotation", {}),
    "annotation.human.task_description present",
)

print("== GR00T loader ==")
sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402,F401  registers NEW_EMBODIMENT on import

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402

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
