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

Shared because these were byte-identical copies in two scorers, and a statistic that two
pipelines report under the same name should be the same code. Reported, never gated: how a
policy actually performs is judged by watching the arm, not by reading a number.

Not shared, deliberately: how a model produces trajectories, what it calls its dimensions, and
everything about its checkpoint. Those belong to the model.
"""

from __future__ import annotations

import statistics

import numpy as np


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
