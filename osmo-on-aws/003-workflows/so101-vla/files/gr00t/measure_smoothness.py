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
import statistics
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


def reversal_rate(series: list[float]) -> float:
    """Fraction of interior samples where the direction of travel flips.

    The statistic the hardware run needed and stage 03 never computed. Scale-free, so a
    smooth slow trajectory and a smooth fast one both score near zero, and only
    oscillation scores high.
    """
    if len(series) < 3:
        return float("nan")
    flips = sum(
        1
        for i in range(1, len(series) - 1)
        if (series[i] - series[i - 1]) * (series[i + 1] - series[i]) < 0
    )
    return flips / (len(series) - 2)


def summarize(chunks: list[np.ndarray]) -> dict:
    """Per-joint smoothness over a list of (T, 6) predicted chunks.

    Computed **within** each chunk. The hardware trace showed within-chunk and
    at-boundary reversal rates were the same, so the chunk itself is where the jitter
    lives and a per-chunk statistic is the honest one.
    """
    out = {}
    for j, name in enumerate(JOINTS):
        rates, steps = [], []
        for chunk in chunks:
            series = [float(v) for v in chunk[:, j]]
            rates.append(reversal_rate(series))
            steps += [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
        out[name] = {
            "reversals": round(statistics.fmean(r for r in rates if r == r), 4),
            "step_p50": round(statistics.median(steps), 4),
        }
    return out


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
    truth_smooth = summarize([truth])
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

        smooth = summarize(chunks)
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
