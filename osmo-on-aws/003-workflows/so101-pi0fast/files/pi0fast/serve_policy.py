#!/usr/bin/env python3
"""Serve a LeRobot pi0-FAST checkpoint over ZMQ, on the observation contract the arm speaks.

Written because **neither shipped policy server can serve this checkpoint**: openpi's
`WebsocketPolicyServer` loads openpi's own checkpoint layout, and LeRobot's
`async_inference/policy_server.py` lists the policies it supports and `pi0_fast` is not among
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


class Pi0FastHandler:
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
        self.state_key = "observation.state"
        print(
            f"loaded {cfg.type} on {self.device}: chunk={cfg.chunk_size} "
            f"n_action_steps={cfg.n_action_steps} dtype={cfg.dtype} "
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
        # The contract names cameras "front" and "wrist"; the dataset names them
        # observation.images.<name>. Mapped by suffix rather than by position, because
        # swapping the two cameras is a silent failure the policy cannot report.
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
            batch[key] = (tensor / 255.0).unsqueeze(0)
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
    p.add_argument("--repo-id", default="local/so101-pi0fast")
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

    from zmq_transport import PolicyServer

    handler = Pi0FastHandler(args.checkpoint, args.dataset_root, args.repo_id, args.jitter_deg)
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
