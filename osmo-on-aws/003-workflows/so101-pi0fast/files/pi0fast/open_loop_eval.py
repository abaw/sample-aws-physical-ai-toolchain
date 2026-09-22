#!/usr/bin/env python3
"""Score a pi0-FAST checkpoint against ground truth, without a robot.

Stage 03's gate. An under-trained policy, a wrong joint order and missing normalisation
statistics all look identical on real hardware — an arm that moves wrongly — and this tells
them apart for a few GPU-minutes.

Nothing upstream does this for `pi0_fast`, so unlike the GR00T pipeline (which drives
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


# Shared with every other model's scorer. The degeneracy test in particular must not
# diverge: it is what stops a policy that does not move from passing a smoothness gate.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from trajectory_metrics import degenerate_dimensions, summarize  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="a pretrained_model/ directory")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--repo-id", default="local/so101-pi0fast")
    p.add_argument("--traj-ids", required=True, help="space-separated episode indices")
    p.add_argument(
        "--execution-horizon",
        type=int,
        default=16,
        help="Actions consumed per chunk before re-planning. Scoring must use the same "
        "value the robot will, or it measures a trajectory nobody executes.",
    )
    p.add_argument("--max-steps", type=int, default=400)
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
        length = min(end - start, args.max_steps)

        predicted, truth = [], []
        policy.reset()
        for offset in range(0, length - args.execution_horizon, args.execution_horizon):
            item = dataset[start + offset]
            batch = {
                k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
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

    # A constant prediction scores ZERO reversals, which reads as perfect smoothness, so the
    # smoothness gate can be passed by a policy that does not move at all. The test lives in
    # the shared module because every model needs it and a drifting second copy would stop
    # catching exactly the case it exists for.
    degenerate = []
    for row in scored:
        degenerate += degenerate_dimensions(
            row["predicted"], row["ground_truth"], traj_id=row["traj_id"]
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
        "degenerate_prediction": bool(degenerate),
        "degenerate_dimensions": degenerate,
        "per_trajectory": scored,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("\n=== summary ===", flush=True)
    for key in (
        "complete", "mse", "mae", "degenerate_prediction",
        "worst_reversals_predicted", "worst_reversals_ground_truth", "reversals_warning",
    ):
        print(f"  {key}: {metrics[key]}", flush=True)
    if degenerate:
        worst = min(degenerate, key=lambda d: d["predicted_travel"] / d["ground_truth_travel"])
        print(
            f"\nWARNING: the prediction barely moves on {len(degenerate)} "
            f"trajectory/dimension pair(s) — worst is {worst['dim']} on trajectory "
            f"{worst['traj_id']}, travelling {worst['predicted_travel']} against the ground "
            f"truth's {worst['ground_truth_travel']}. A constant output scores zero "
            f"reversals, so read the reversal rate above as meaningless here. This is what "
            f"an under-trained checkpoint looks like: it has learned the mean pose.",
            file=sys.stderr,
            flush=True,
        )
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
