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
