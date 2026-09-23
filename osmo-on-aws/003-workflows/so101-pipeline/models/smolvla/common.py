# SPDX-License-Identifier: Apache-2.0
"""The camera-name mapping SmolVLA needs, defined once for every stage that needs it.

`lerobot/smolvla_base` was pretrained with its image features named `observation.images.camera1`
through `camera3`. A recording names them whatever it named them — `front`, `wrist`. Fine-tuning
with `--policy.path` loads the pretrained config, expected feature names included, so LeRobot
refuses the dataset outright:

    Missing features: ['observation.images.camera1', ...]
    Extra features:   ['observation.images.front', 'observation.images.wrist']

This is the same indirection contract B calls a camera's three identities, in a third spelling: the
dataset feature key and the model's internal key need not match, and every model has its own way of
saying so. GR00T uses `modality.json`'s `original_key`; LeRobot uses `--rename_map`.

**This module exists so training and serving cannot disagree about it.** The mapping is arbitrary —
`camera1` carries no meaning — so what matters is only that it is the SAME at both ends. A second
copy that drifted would train on one assignment and serve on another, and the arm would be driven
from frames the policy associates with a different viewpoint. Nothing downstream could see it.

`common.py` is injected into every stage for exactly this kind of thing.
"""

from __future__ import annotations

IMAGE_PREFIX = "observation.images."


def rename_map(dataset_image_keys: list[str]) -> dict[str, str]:
    """Dataset image features -> the slots this model was pretrained with.

    Sorted, so the assignment is deterministic and reproducible across stages and runs. Sorting is
    a positional mapping, which contract B warns against for good reason — but here there is no
    alternative: the target slots are numbered, not named, so no semantic match is possible. What
    the warning protects against is the mapping DIFFERING between the two ends, and a single sorted
    rule in a single file is what rules that out.
    """
    ordered = sorted(dataset_image_keys)
    return {key: f"{IMAGE_PREFIX}camera{index}" for index, key in enumerate(ordered, start=1)}


def image_keys_from_info(info: dict) -> list[str]:
    """The dataset's own image features, which are the authority on what cameras exist."""
    return [k for k in info.get("features", {}) if k.startswith(IMAGE_PREFIX)]
