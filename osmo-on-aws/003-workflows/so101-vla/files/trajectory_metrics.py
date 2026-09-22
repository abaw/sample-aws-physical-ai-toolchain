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
"""Trajectory statistics shared by every model's open-loop scoring.

Shared because the **degeneracy** test below is subtle and must not diverge between models:
it is the check that stops a policy which does not move from passing a smoothness gate, and a
second copy that drifts would silently stop catching that. The reversal rate travels with it
because the two are only meaningful together.

Not shared, deliberately: how a model produces trajectories, what it calls its dimensions, and
everything about its checkpoint. Those belong to the model.
"""

from __future__ import annotations

import statistics

import numpy as np

# A dimension whose ground truth barely moves cannot say anything about whether a prediction
# moved, so it is excluded from the degeneracy test rather than counted as a pass.
MIN_GROUND_TRUTH_TRAVEL = 1.0

# Below this fraction of the ground truth's own travel, a prediction is not tracking the task.
# Relative rather than absolute because what counts as "barely moving" depends on the task.
DEGENERATE_TRAVEL_FRACTION = 0.05


def reversal_rate(series: list[float]) -> float:
    """Fraction of interior samples where the direction of travel flips.

    Scale-free, so a smooth slow trajectory and a smooth fast one both score near zero, and
    only oscillation scores high — amplitude alone cannot tell traversing the workspace apart
    from vibrating in place, and it is the latter that overheats a servo.

    For calibration against known signals: a clean monotonic ramp scores 0, a random walk
    about 0.5, and a per-sample sawtooth 1.0. Teleoperated recordings score under 1%, which is
    what makes the ground-truth figure a check on the metric rather than decoration.
    """
    if len(series) < 3:
        return float("nan")
    flips = sum(
        1
        for i in range(1, len(series) - 1)
        if (series[i] - series[i - 1]) * (series[i + 1] - series[i]) < 0
    )
    return flips / (len(series) - 2)


def summarize(chunks: list[np.ndarray], names: list[str]) -> dict:
    """Per-dimension statistics over a list of (T, D) arrays.

    **Reversals are computed within each chunk and then averaged**, never over the
    concatenation. A hardware trace showed within-chunk and at-boundary reversal rates were
    the same, so the chunk is where the jitter lives, and splicing chunks together would
    invent direction changes at every seam.

    **Travel is computed over the concatenation**, because it answers a different question —
    did this dimension go anywhere over the whole trajectory — and a per-chunk maximum would
    understate it.

    A single trajectory is a list of one chunk, in which case both reduce to the obvious
    thing: this is what keeps one model's recorded numbers unchanged by the sharing.
    """
    if not chunks:
        raise ValueError("no chunks to summarize")

    joined = np.concatenate(chunks, axis=0)
    out = {}
    for index, name in enumerate(names):
        rates, steps = [], []
        for chunk in chunks:
            series = [float(v) for v in chunk[:, index]]
            rates.append(reversal_rate(series))
            steps += [abs(series[i] - series[i - 1]) for i in range(1, len(series))]

        # Chunks shorter than three samples yield nan, and averaging over none of them would
        # raise rather than report "unknown".
        finite = [r for r in rates if r == r]
        whole = [float(v) for v in joined[:, index]]
        out[name] = {
            "reversals": round(statistics.fmean(finite), 4) if finite else float("nan"),
            "step_p50": round(statistics.median(steps), 4) if steps else None,
            "min": round(min(whole), 3),
            "max": round(max(whole), 3),
            # Reported beside the reversal rate because zero reversals is ambiguous without
            # it: a constant output and a perfectly smooth sweep both score 0.
            "travel": round(max(whole) - min(whole), 3),
        }
    return out


def degenerate_dimensions(
    predicted: dict,
    ground_truth: dict,
    *,
    traj_id: int | None = None,
) -> list[dict]:
    """Dimensions where the prediction barely moved while the ground truth did.

    This is the check that makes a smoothness score trustworthy. An under-trained checkpoint
    emits the dataset's mean pose whatever it is shown; a constant output scores **zero**
    reversals, which reads as perfect smoothness. Without this test, the least useful possible
    policy earns the best possible smoothness number and passes the gate.

    Measured against the ground truth's own travel rather than an absolute threshold, and
    dimensions the ground truth barely exercises are skipped rather than passed, because they
    carry no evidence either way.
    """
    findings = []
    for name, truth in ground_truth.items():
        if name not in predicted:
            continue
        truth_travel = truth.get("travel")
        predicted_travel = predicted[name].get("travel")
        if truth_travel is None or predicted_travel is None:
            continue
        if truth_travel <= MIN_GROUND_TRUTH_TRAVEL:
            continue
        if predicted_travel < DEGENERATE_TRAVEL_FRACTION * truth_travel:
            # traj_id first so the serialized order matches what callers already publish.
            finding = {} if traj_id is None else {"traj_id": traj_id}
            finding.update(
                dim=name,
                predicted_travel=predicted_travel,
                ground_truth_travel=truth_travel,
            )
            findings.append(finding)
    return findings
