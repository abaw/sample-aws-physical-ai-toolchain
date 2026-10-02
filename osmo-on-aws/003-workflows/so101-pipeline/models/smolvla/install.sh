#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install LeRobot with the SmolVLA policy.  $1 = stage name.  SOURCED, not executed.
#
# The stage argument is ignored here: every stage needs the same LeRobot venv, because even
# dataset conversion runs LeRobot's own converter script.
set -euo pipefail
STAGE="${1:?install.sh needs the stage name}"
echo "[smolvla] install for stage: ${STAGE}"

# The model's own system dependencies, installed by the model. The pipeline does not know that
# this toolchain needs curl to fetch itself, ffmpeg for torchcodec (without libav* its library
# fails to load and LeRobot silently falls back to another decoder, so training and serving would
# see frames decoded differently), or uv to resolve the venv.
if ! command -v curl >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq curl ffmpeg
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

LEROBOT_TAG="${LEROBOT_TAG:-v0.6.1}"
LEROBOT_DIR="${LEROBOT_DIR:-/tmp/lerobot}"

# A tarball, not a clone: this image has no git.
mkdir -p "${LEROBOT_DIR}"
curl -fsSL "https://github.com/huggingface/lerobot/archive/refs/tags/${LEROBOT_TAG}.tar.gz" \
  | tar -xz --strip-components=1 -C "${LEROBOT_DIR}"

cd "${LEROBOT_DIR}"

# One resolve, not several: separate pip installs let a later one silently downgrade an
# earlier one's pins.
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[smolvla,scipy-dep,dataset,training]"

python - <<'PY'
import importlib, torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
c = SmolVLAConfig()
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
print(f"transformers {importlib.import_module('transformers').__version__}")
# Printed, not asserted: upstream defaults change, and the log should say so.
print(
    f"smolvla defaults: chunk={c.chunk_size} n_action_steps={c.n_action_steps} "
    f"max_action_dim={c.max_action_dim} vlm={c.vlm_model_name} "
    f"freeze_vision_encoder={c.freeze_vision_encoder} load_vlm_weights={c.load_vlm_weights} "
    f"lr={c.optimizer_lr} warmup={c.scheduler_warmup_steps}"
)
PY

export LEROBOT_DIR
echo "LEROBOT_DIR=${LEROBOT_DIR} (venv on PATH)"
# Serving needs the transport's dependencies in this venv; no other stage opens a socket. This
# is the one thing the stage argument gates.
if [ "${STAGE}" = "serve" ]; then
  uv pip install "pyzmq==${PYZMQ_VERSION:-27.0.1}" "msgpack==${MSGPACK_VERSION:-1.1.0}"
  echo "[smolvla] transport dependencies installed for serving"
fi

echo "[smolvla] toolchain ready for stage: ${STAGE}"

# A checkpoint argument may be a pretrained_model/ directory or a training output holding several
# steps. `export -f` is required: the stage scripts are EXECUTED, so they inherit exported
# functions but not plain ones.
smolvla_resolve_checkpoint() {
  local given="${1:?smolvla_resolve_checkpoint needs a path}"
  if [ -f "${given}/config.json" ]; then
    echo "${given}"
    return 0
  fi
  local found
  # -V because a lexical sort puts 090000 before 100000 only by luck of zero padding.
  found="$(find "${given}" -type d -name pretrained_model | sort -V | tail -1)"
  if [ -z "${found}" ]; then
    echo "ERROR: no pretrained_model/ directory under ${given}" >&2
    find "${given}" -maxdepth 4 -type d >&2
    return 1
  fi
  echo "${found}"
}
export -f smolvla_resolve_checkpoint

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
MODEL_FILE_EOF

case "${STAGE}" in
  prepare)
cat > "${MODEL_FILES}/write_episode_stats.py" <<'MODEL_FILE_EOF'
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
MODEL_FILE_EOF

    ;;
  eval)
cat > "${MODEL_FILES}/score_offline.py" <<'MODEL_FILE_EOF'
#!/usr/bin/env python3
"""Score a SmolVLA checkpoint against ground truth, without a robot.

Stage 03's gate. An under-trained policy, a wrong joint order and missing normalisation
statistics all look identical on real hardware — an arm that moves wrongly — and this tells
them apart for a few GPU-minutes.

Nothing upstream does this for `smolvla`, so unlike the GR00T pipeline (which drives
upstream's own eval and then scrapes the numbers back out of its log, because upstream only
prints them) this writes `metrics.json` itself. The artifact therefore cannot disagree with
the log, and there is no regex against another project's output format to break.

Two families of number, because they answer different questions and only one of them is
visible in the plots:

* **MSE / MAE**, overall and per action dimension. Per-dimension because one bad joint hides
  inside a good average, and a wrong joint *order* appears as several bad dimensions at once.
* **Direction-reversal rate**, for the prediction and for the ground truth it imitates. MSE
  cannot see jitter: a zero-mean oscillation on a broadly correct trajectory scores well and
  cooks the smallest servo. The definition is copied verbatim from the GR00T pipeline's
  `measure_smoothness.py` so the two models' numbers are directly comparable.
"""

import argparse
import json
import pathlib
import statistics
import sys

import numpy as np
import torch


# Shared with every other model's scorer, so a statistic two pipelines report under the
# same name is the same code.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from trajectory_metrics import summarize  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="a pretrained_model/ directory")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", default="local/so101-smolvla")
    p.add_argument("--traj-ids", required=True, help="space-separated episode indices")
    p.add_argument(
        "--execution-horizon",
        type=int,
        default=16,
        help="Actions consumed per chunk before re-planning. Scoring must use the same "
        "value the robot will, or it measures a trajectory nobody executes.",
    )
    p.add_argument(
        "--frames-per-trajectory",
        type=int,
        default=400,
        help="Frames of each trajectory to replay, capped by the episode length.",
    )
    p.add_argument("--warn-reversals", type=float, default=0.25)
    p.add_argument("--output-dir", required=True)
    return p


def main() -> int:  # noqa: PLR0915 - a linear scoring pass, ordered by trajectory
    args = build_parser().parse_args()
    output = pathlib.Path(args.output_dir)
    (output / "plots").mkdir(parents=True, exist_ok=True)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.device = device

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=args.checkpoint, dataset_stats=dataset.meta.stats
    )

    # The same camera-slot mapping training used, from the same function, so scoring measures the
    # model that will be served rather than one fed a differently-assigned rig. See common.py.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from common import rename_map  # noqa: PLC0415 - beside this file, injected as a slot

    image_keys = [k for k in dataset.meta.features if k.startswith("observation.images")]
    RENAME = rename_map(image_keys)
    print(f"rename_map: {RENAME}", flush=True)

    dim = dataset.meta.features[ACTION]["shape"][0]
    names = dataset.meta.features[ACTION].get("names") or [f"dim_{i}" for i in range(dim)]
    if args.execution_horizon > cfg.n_action_steps:
        sys.exit(
            f"ERROR: --execution-horizon {args.execution_horizon} exceeds the policy's "
            f"n_action_steps {cfg.n_action_steps}"
        )

    requested = [int(t) for t in args.traj_ids.split()]
    print(
        f"device={device} action_dim={dim} names={names} "
        f"chunk={cfg.chunk_size} execution_horizon={args.execution_horizon}",
        flush=True,
    )

    per_traj = {}
    for traj_id in requested:
        if traj_id < 0 or traj_id >= dataset.meta.total_episodes:
            # Skipped rather than fatal, but recorded — see `complete` below. A silent skip
            # is how a typo shrinks the sample an average is taken over.
            print(
                f"WARNING: trajectory {traj_id} out of range "
                f"(dataset has {dataset.meta.total_episodes})",
                file=sys.stderr,
                flush=True,
            )
            continue

        start = dataset.meta.episodes["dataset_from_index"][traj_id]
        end = dataset.meta.episodes["dataset_to_index"][traj_id]
        length = min(end - start, args.frames_per_trajectory)

        predicted, truth = [], []
        policy.reset()
        for offset in range(0, length - args.execution_horizon, args.execution_horizon):
            item = dataset[start + offset]
            batch = {
                RENAME.get(k, k): (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in item.items()
            }
            chunk = policy.predict_action_chunk(preprocessor(batch))
            chunk = postprocessor(chunk)[0].float().cpu().numpy()  # (chunk, dim)

            for step in range(args.execution_horizon):
                if offset + step >= length:
                    break
                predicted.append(chunk[step, :dim])
                truth.append(dataset[start + offset + step][ACTION].float().numpy()[:dim])

        if not predicted:
            print(f"WARNING: trajectory {traj_id} produced no steps", file=sys.stderr)
            continue

        predicted_array = np.stack(predicted)
        truth_array = np.stack(truth)
        error = predicted_array - truth_array
        per_traj[traj_id] = {
            "traj_id": traj_id,
            "steps": int(predicted_array.shape[0]),
            "mse": round(float((error**2).mean()), 4),
            "mae": round(float(np.abs(error).mean()), 4),
            "mse_by_dim": {
                n: round(float((error[:, i] ** 2).mean()), 4) for i, n in enumerate(names)
            },
            "mae_by_dim": {
                n: round(float(np.abs(error[:, i]).mean()), 4) for i, n in enumerate(names)
            },
            "predicted": summarize([predicted_array], names),
            "ground_truth": summarize([truth_array], names),
        }
        row = per_traj[traj_id]
        worst = max(v["reversals"] for v in row["predicted"].values())
        truth_worst = max(v["reversals"] for v in row["ground_truth"].values())
        print(
            f"traj {traj_id}: {row['steps']} steps  MSE {row['mse']}  MAE {row['mae']}  "
            f"worst reversals predicted {worst:.1%} vs truth {truth_worst:.1%}",
            flush=True,
        )

        # One plot per trajectory, per dimension. The numbers are the gate; the plots are how
        # a person sees *where* it went wrong.
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(dim, 1, figsize=(11, 2.0 * dim), sharex=True)
            for index, name in enumerate(names):
                ax = axes[index] if dim > 1 else axes
                ax.plot(truth_array[:, index], label="ground truth", linewidth=1.2)
                ax.plot(predicted_array[:, index], label="predicted", linewidth=1.0, alpha=0.8)
                ax.set_ylabel(name, fontsize=8)
                if index == 0:
                    ax.legend(fontsize=8)
            axes[-1].set_xlabel("step") if dim > 1 else axes.set_xlabel("step")
            fig.suptitle(f"trajectory {traj_id} — MSE {row['mse']}, MAE {row['mae']}")
            fig.tight_layout()
            fig.savefig(output / "plots" / f"traj_{traj_id:04d}.png", dpi=110)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - plots are not the gate
            print(f"WARNING: plotting trajectory {traj_id} failed: {exc}", file=sys.stderr)

    evaluated = sorted(per_traj)
    missing = [t for t in requested if t not in evaluated]
    scored = list(per_traj.values())

    def mean_of(key: str) -> float | None:
        values = [r[key] for r in scored]
        return round(statistics.fmean(values), 4) if values else None

    worst_predicted = (
        round(max(max(v["reversals"] for v in r["predicted"].values()) for r in scored), 4)
        if scored else None
    )
    worst_truth = (
        round(max(max(v["reversals"] for v in r["ground_truth"].values()) for r in scored), 4)
        if scored else None
    )

    metrics = {
        "checkpoint": args.checkpoint,
        "execution_horizon": args.execution_horizon,
        "action_dim": dim,
        "action_names": names,
        # Not the same question as whether the task exited 0: out-of-range IDs are skipped
        # with a warning, so without this a typo quietly shrinks the sample.
        "complete": not missing,
        "requested_trajectories": requested,
        "evaluated_trajectories": evaluated,
        "missing_trajectories": missing,
        "mse": mean_of("mse"),
        "mae": mean_of("mae"),
        "worst_reversals_predicted": worst_predicted,
        "worst_reversals_ground_truth": worst_truth,
        # The ground-truth figure is the reference, not decoration: teleop recordings score
        # under 1%, so if it does not, the metric or the dataset is wrong rather than the
        # policy.
        "reversals_warn_threshold": args.warn_reversals,
        "reversals_warning": bool(
            worst_predicted is not None and worst_predicted > args.warn_reversals
        ),
        # Check this BEFORE reading the reversal rate. True means the policy is barely moving
        # the joint, so its zero reversal rate means nothing.
        "per_trajectory": scored,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("\n=== summary ===", flush=True)
    for key in (
        "complete", "mse", "mae",
        "worst_reversals_predicted", "worst_reversals_ground_truth", "reversals_warning",
    ):
        print(f"  {key}: {metrics[key]}", flush=True)
    if missing:
        print(
            f"\nERROR: {len(missing)} requested trajectory/ies never scored: {missing}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if metrics["reversals_warning"]:
        print(
            f"\nWARNING: worst predicted reversal rate {worst_predicted:.1%} exceeds "
            f"{args.warn_reversals:.0%}. The policy is emitting per-step increments with "
            f"little trend; on hardware this vibrates a joint rather than moving it, and it "
            f"is what overheats the smallest servo. Ground truth for comparison: "
            f"{worst_truth:.1%}.",
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
MODEL_FILE_EOF

    ;;
  serve)
cat > "${MODEL_FILES}/serve_policy.py" <<'MODEL_FILE_EOF'
#!/usr/bin/env python3
"""Serve a LeRobot SmolVLA checkpoint over ZMQ, on the observation contract the arm speaks.

Written because **neither shipped policy server can serve this checkpoint**: openpi's
`WebsocketPolicyServer` loads openpi's own checkpoint layout, and LeRobot's
`async_inference/policy_server.py` lists the policies it supports and `smolvla` is not among
them. That is a finding rather than an inconvenience — serving was expected to be the cheap
part of adding a second model and it was not.

The wire contract is deliberately the one the robot client already speaks, so that client is
reused unchanged rather than copied. It is shaped like the reference model's:

    observation  {"video":    {"front": (1,1,H,W,3) uint8, "wrist": ...},
                  "state":    {"single_arm": (1,1,5) float32, "gripper": (1,1,1) float32},
                  "language": {"annotation.human.task_description": [str]}}
    action       {"single_arm": (1,T,5), "gripper": (1,T,1)}

Adapting that to LeRobot's flat batch is this file's job. Putting the adapter in the server
rather than in the client is what keeps the arm-safety code single-sourced: the contract is
the pipeline's interface, and each model's server meets it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

# Joint order is load-bearing: the order the dataset was recorded in, the order the action
# head emits, and the order the servos are addressed in.
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LANGUAGE_KEY = "annotation.human.task_description"

# A test affordance, not a feature. Bounded here so a mistyped --jitter-deg cannot turn a
# small tracking check into a large movement; the client's envelope and per-step delta limits
# bound it again independently.
MAX_JITTER_DEG = 5.0


def _strip_leading(array: np.ndarray, count: int = 2) -> np.ndarray:
    """Drop the batch and time dimensions the client adds."""
    for _ in range(count):
        if array.ndim > 1 and array.shape[0] == 1:
            array = array[0]
    return array


class SmolVLAHandler:
    """Turns one contract observation into one contract action chunk."""

    def __init__(self, checkpoint: str, dataset_root: str, repo_id: str, jitter_deg: float):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.jitter_deg = jitter_deg
        self.rng = np.random.default_rng(0)

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = self.device
        self.cfg = cfg

        # The dataset is loaded for its statistics, not its frames: the preprocessor
        # normalises state and un-normalises actions with them, and a policy served with
        # different statistics than it trained on drives confidently to the wrong pose.
        dataset = LeRobotDataset(repo_id, root=dataset_root)
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint, dataset_stats=dataset.meta.stats
        )
        self.image_keys = [k for k in dataset.meta.features if k.startswith("observation.images")]
        # The SAME mapping training used, from the same function, so the two cannot disagree about
        # which viewpoint a numbered slot holds. See common.py.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from common import rename_map  # noqa: PLC0415 - beside this file, injected as a slot

        self.rename = rename_map(self.image_keys)
        self.state_key = "observation.state"
        print(
            f"loaded {cfg.type} on {self.device}: chunk={cfg.chunk_size} "
            # SmolVLAConfig has no dtype field, hence the getattr default.
            f"n_action_steps={cfg.n_action_steps} dtype={getattr(cfg, 'dtype', 'n/a')} "
            f"image_keys={self.image_keys} jitter_deg={jitter_deg}",
            flush=True,
        )
        if jitter_deg:
            print(
                f"WARNING: jitter is ON at +/-{jitter_deg} deg. Actions are perturbed, so "
                f"any trace from this run is NOT the policy's output.",
                file=sys.stderr,
                flush=True,
            )

    def __call__(self, observation: dict) -> tuple[dict, dict]:
        start = time.perf_counter()

        video = observation["video"]
        state = _strip_leading(np.asarray(observation["state"]["single_arm"], dtype=np.float32))
        gripper = _strip_leading(np.asarray(observation["state"]["gripper"], dtype=np.float32))
        full_state = np.concatenate([np.atleast_1d(state), np.atleast_1d(gripper)])
        if full_state.shape[0] != len(JOINTS):
            raise ValueError(
                f"state has {full_state.shape[0]} values, expected {len(JOINTS)} "
                f"({JOINTS}) — check the client's joint order"
            )

        # The client adds a batch and a time dimension by wrapping non-arrays in a list
        # ONCE EACH, so the instruction arrives as [["..."]]. Unwrapped in a loop rather than
        # with two fixed indexes: the preprocessor iterates this and calls .strip() on each
        # element, so a residual list reads as `'list' object has no attribute 'strip'` from
        # inside the tokenizer, which says nothing about where it came from.
        task = observation["language"][LANGUAGE_KEY]
        while isinstance(task, (list, tuple)) and len(task) == 1:
            task = task[0]
        if isinstance(task, bytes):
            # msgpack decodes a str key's value as str, but a bytes value stays bytes and
            # would reach the tokenizer as b"..." and tokenize as nonsense.
            task = task.decode()
        if not isinstance(task, str):
            raise ValueError(
                f"language instruction decoded as {type(task).__name__}, expected str: {task!r}"
            )

        batch: dict = {self.state_key: torch.from_numpy(full_state).unsqueeze(0)}
        # Cameras are named by the dataset's image-feature suffix — the dataset names them
        # observation.images.<name> — and matched by name rather than by position, because
        # swapping two cameras is a silent failure the policy cannot report.
        #
        # Both directions are errors. A missing camera is caught in the loop below; an
        # *extra* one has to be caught here, because iterating the expected keys would
        # simply drop it and serve an action computed from a rig the policy never saw.
        expected = {key.rsplit(".", 1)[-1] for key in self.image_keys}
        unexpected = sorted(set(video) - expected)
        if unexpected:
            raise ValueError(
                f"observation carries camera(s) {unexpected} the checkpoint was not trained "
                f"on; it expects exactly {sorted(expected)}. Serving anyway would drop them "
                f"silently and drive from an input the policy never saw."
            )
        for key in self.image_keys:
            name = key.rsplit(".", 1)[-1]
            if name not in video:
                raise ValueError(
                    f"observation has no camera {name!r}; got {sorted(video)} and the "
                    f"checkpoint expects {self.image_keys}"
                )
            frame = _strip_leading(np.asarray(video[name]))
            if frame.ndim != 3:
                raise ValueError(f"camera {name!r} has shape {frame.shape}, expected (H,W,3)")
            # HWC uint8 -> CHW float in [0,1], which is what LeRobot's dataset yields.
            # .copy() because the frame is a read-only view over the msgpack buffer, and
            # torch.from_numpy on a non-writable array warns and is undefined if written to.
            tensor = torch.from_numpy(np.ascontiguousarray(frame).copy()).permute(2, 0, 1)
            tensor = tensor.float()
            batch[self.rename[key]] = (tensor / 255.0).unsqueeze(0)
        batch["task"] = [task]

        batch = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
        }
        with torch.no_grad():
            chunk = self.policy.predict_action_chunk(self.preprocessor(batch))
            chunk = self.postprocessor(chunk)
        actions = chunk[0].float().cpu().numpy()[:, : len(JOINTS)]  # (T, 6)

        if self.jitter_deg:
            # Uniform, zero-mean, per joint per step. Deliberately not a trajectory: this
            # proves the arm tracks a *changing* command, which a constant one cannot.
            actions = actions + self.rng.uniform(
                -self.jitter_deg, self.jitter_deg, size=actions.shape
            ).astype(np.float32)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (
            {
                "single_arm": actions[np.newaxis, :, :5].astype(np.float32),
                "gripper": actions[np.newaxis, :, 5:6].astype(np.float32),
            },
            # Reported so a trace cannot be mistaken for unperturbed policy output.
            {"infer_ms": round(elapsed_ms, 2), "jitter_deg": self.jitter_deg},
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="a pretrained_model/ directory")
    p.add_argument("--dataset-root", required=True, help="for normalisation statistics")
    p.add_argument("--repo-id", default="local/so101-smolvla")
    p.add_argument("--host", default="0.0.0.0")  # noqa: S104 - one pod, cluster-internal
    p.add_argument("--port", type=int, default=5555)
    p.add_argument(
        "--jitter-deg",
        type=float,
        default=0.0,
        help="TEST ONLY. Add uniform noise of this magnitude to every action, so a "
        "constant-output checkpoint still makes the arm move and its tracking can be "
        f"checked. Refused above {MAX_JITTER_DEG} deg.",
    )
    args = p.parse_args()

    if args.jitter_deg < 0 or args.jitter_deg > MAX_JITTER_DEG:
        # Before the model loads and before the socket opens: a mistyped magnitude must not
        # become a movement.
        sys.exit(
            f"ERROR: --jitter-deg {args.jitter_deg} is outside 0..{MAX_JITTER_DEG}. This is a "
            f"tracking test, not a way to drive the arm."
        )

    # transport.py has a fixed name, so the server and the robot client import the same module.
    from transport import PolicyServer

    handler = SmolVLAHandler(args.checkpoint, args.dataset_root, args.repo_id, args.jitter_deg)
    server = PolicyServer(handler, host=args.host, port=args.port)
    print(f"serving on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
MODEL_FILE_EOF

    ;;
esac

echo "[smolvla] model files in ${MODEL_FILES}: $(ls "${MODEL_FILES}" | tr '\n' ' ')"
