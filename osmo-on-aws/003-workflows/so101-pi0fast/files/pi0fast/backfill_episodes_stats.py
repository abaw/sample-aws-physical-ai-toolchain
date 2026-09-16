#!/usr/bin/env python3
"""Write the `meta/episodes_stats.jsonl` that LeRobot's v2.1 -> v3.0 converter demands.

Our recording declares `codebase_version: v2.1` but carries the **v2.0** metadata layout:
an aggregate `meta/stats.json` and no per-episode `meta/episodes_stats.jsonl`. The v3
converter validates the version string, believes it, and then dies in
`legacy_load_episodes_stats` on a file that was never written. LeRobot used to ship a
v2.0 -> v2.1 script that did exactly this backfill; it is gone as of v0.6.1, so this
stands in for it.

Stats are computed with LeRobot's own `get_feature_stats`, not reimplemented, so the shapes
and the quantile set match what the converter and `aggregate_stats` expect by construction.

Video features get shape-correct placeholders rather than measured values, and that is a
deliberate, checked decision rather than a shortcut:

* `PI0FastConfig.normalization_mapping` sets `VISUAL: NormalizationMode.IDENTITY`, so the
  policy does not normalise images at all and never reads these numbers.
* `DatasetConfig.use_imagenet_stats` defaults to True, and the dataset factory *assigns into*
  `dataset.meta.stats[key]["mean"]` and `["std"]` for every camera key. It indexes rather
  than inserts, so the key must EXIST or construction raises `KeyError` — which is why the
  video entries cannot simply be omitted. Whatever mean/std we write is then overwritten.
* `min`/`max` of 0 and 1 are the true range of RGB after the /255 rescale the real
  computation would apply, so they are not invented values, just uninformative ones.

Measuring them properly would mean decoding 43 episodes across two cameras to produce
numbers that provably reach nothing. If a future policy sets VISUAL to MEAN_STD or MIN_MAX,
this becomes wrong and must be replaced with a real per-episode computation over sampled
frames. The numeric features that *do* drive normalisation — state and action, both
MEAN_STD — are computed per episode from the parquet, which is the part that has to be right.
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats

IMAGE_DTYPES = {"image", "video"}
SKIP_DTYPES = {"string", "language"}


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _placeholder_image_stats(frame_count: int) -> dict:
    """Schema-shaped stats for a video feature. See the module docstring for why.

    `_validate_stat_value` requires (3,1,1) or (1,1,1) for any key containing "image", and
    (1,) for `count`, so the shapes are load-bearing even though the values are not.
    """
    per_channel = lambda v: np.full((3, 1, 1), float(v), dtype=np.float32)  # noqa: E731
    stats = {
        "min": per_channel(0.0),
        "max": per_channel(1.0),
        # Overwritten by IMAGENET_STATS at dataset construction; written so the key exists.
        "mean": per_channel(0.5),
        "std": per_channel(0.25),
        "count": np.array([frame_count]),
    }
    for quantile in DEFAULT_QUANTILES:
        stats[f"q{int(quantile * 100):02d}"] = per_channel(0.5)
    return stats


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    out = root / "meta" / "episodes_stats.jsonl"
    if out.exists():
        print(f"{out} already exists; nothing to do")
        return 0

    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]

    # Episode order comes from episodes.jsonl, because the converter zips these lists
    # positionally against its own scan of the data directory.
    episodes = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    episodes.sort(key=lambda e: e["episode_index"])
    print(f"{len(episodes)} episodes, {len(features)} features")

    parquets = sorted(root.glob("data/**/episode_*.parquet"))
    by_index = {int(p.stem.split("_")[-1]): p for p in parquets}
    missing = [e["episode_index"] for e in episodes if e["episode_index"] not in by_index]
    if missing:
        sys.exit(f"ERROR: no parquet for episode(s) {missing[:5]} (of {len(missing)})")

    video_keys = set()
    with out.open("w") as handle:
        for episode in episodes:
            index = episode["episode_index"]
            frame = pd.read_parquet(by_index[index])
            stats = {}

            for key, spec in features.items():
                dtype = spec["dtype"]
                if dtype in SKIP_DTYPES:
                    continue

                if dtype in IMAGE_DTYPES:
                    video_keys.add(key)
                    stats[key] = _placeholder_image_stats(len(frame))
                    continue

                if key not in frame.columns:
                    sys.exit(f"ERROR: numeric feature {key} missing from {by_index[index].name}")

                # A parquet cell is either a scalar or a list; np.stack normalises both to
                # (frames, dims). keepdims mirrors compute_episode_stats: scalar columns
                # keep their reduced axis so `count` and friends stay 1-D.
                array = np.stack([np.atleast_1d(v) for v in frame[key].to_numpy()])
                if array.shape[1] == 1 and len(spec.get("shape") or [1]) == 1:
                    array = array.reshape(-1)
                stats[key] = get_feature_stats(
                    array.astype(np.float32),
                    axis=0,
                    keepdims=array.ndim == 1,
                    quantile_list=DEFAULT_QUANTILES,
                )

            handle.write(
                json.dumps({"episode_index": index, "stats": _to_jsonable(stats)}) + "\n"
            )

    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"per-episode numeric stats computed from parquet; schema-only placeholder stats "
          f"written for {sorted(video_keys)} (VISUAL normalisation is IDENTITY)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
