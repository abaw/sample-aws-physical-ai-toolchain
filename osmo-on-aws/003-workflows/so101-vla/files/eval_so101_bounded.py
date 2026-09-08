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
"""Closed-loop policy evaluation on a physical SO-101 follower arm.

Three sources, deliberately kept separate:

* **Robot I/O and the observation contract** follow upstream's own real-robot client,
  ``gr00t/eval/real_robot/SO100/eval_so100.py``: the same joint order, the same
  ``single_arm`` (5) / ``gripper`` (1) state split, the same two leading singleton
  dimensions, the same ``robot.send_action`` call. A divergence there is a silent
  wrong-pose bug, so it is a copy, not an interpretation.
* **Loop discipline** is the action-chunking harness shape: consume
  ``min(action_horizon, chunk)`` steps, then re-plan; hold the rate with catch-up so a
  network hiccup does not quietly stretch the trajectory in time.
* **Everything that keeps the arm safe** is this file's own, and every item of it was
  earned by something that happened on this hardware. See ``ArmDriver``.

Units. The arm is driven through LeRobot's ``SO101Follower``, never through raw servo
ticks, because the policy speaks LeRobot's *calibrated* joint space: five arm joints in
degrees about the midpoint of each joint's calibrated range, and the gripper 0-100 over
its own. Reimplementing that mapping is the one bug in this file that would be both
silent and destructive, so it is not reimplemented — it is restated once, in
``_normalized_from_tick``, purely so it can be checked against LeRobot's own answer on
the live arm before any torque is enabled.
"""

import argparse
import dataclasses
import json
import pathlib
import signal
import statistics
import sys
import threading
import time

import numpy as np
import zmq

# Joint order is load-bearing: it is the order the dataset was recorded in, the order
# the checkpoint's action head emits, and the order the servos are addressed in.
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]
GRIPPER = "gripper"
STATE_KEYS = [f"{j}.pos" for j in JOINTS]

# STS3215 has 4096 positions, so the largest index is 4095. LeRobot normalizes against
# resolution - 1, which is the whole reason this constant is not 4096.
RESOLUTION_MINUS_1 = 4095

LANGUAGE_KEY = "annotation.human.task_description"


class Abort(Exception):
    """Raised to unwind through the release path rather than dying where we stand."""


_stop = threading.Event()


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        # A cancelled workflow, a node drain and a Ctrl-C all arrive here. Set the flag
        # *and* raise, so a blocked receive unwinds instead of holding the arm until it
        # times out.
        _stop.set()
        raise Abort(f"received {signal.Signals(signum).name}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, handler)


# ---------------------------------------------------------------------------
# Loop rate
# ---------------------------------------------------------------------------


class RateLimiter:
    """Hold a step rate, and catch up rather than drift when a step runs long.

    A naive ``sleep(period - elapsed)`` silently stretches the trajectory in time
    whenever the loop overruns, which on a real arm means the motion no longer matches
    the timing the policy was trained at. Tracking an absolute next-wakeup instead turns
    an overrun into a *dropped* sleep, and counts it.
    """

    def __init__(self, hz: float):
        self.period = 1.0 / hz
        self.next_wakeup = time.perf_counter()
        self.overruns = 0
        self.worst_overrun_ms = 0.0

    def reset(self) -> None:
        self.next_wakeup = time.perf_counter()

    def sleep(self) -> None:
        self.next_wakeup += self.period
        now = time.perf_counter()
        if now < self.next_wakeup:
            time.sleep(self.next_wakeup - now)
            return
        # Behind schedule. Record it and re-base, so one slow step does not make every
        # later step appear late too.
        self.overruns += 1
        self.worst_overrun_ms = max(self.worst_overrun_ms, (now - self.next_wakeup) * 1000.0)
        self.next_wakeup = now


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def _normalized_from_tick(calibration, joint: str, tick: float) -> float:
    """LeRobot's own normalization, restated so a drift between us can be detected.

    Not used to drive the arm — ``SO101Follower`` does that. Used only to turn the
    calibration's raw ``range_min``/``range_max`` into the units a policy action arrives
    in, so the safety envelope can be expressed in the same space as the actions it
    bounds. ``ArmDriver.verify_units`` checks it against LeRobot on the live arm.
    """
    cal = calibration[joint]
    lo, hi = cal.range_min, cal.range_max
    if joint == GRIPPER:  # MotorNormMode.RANGE_0_100
        return (min(hi, max(lo, tick)) - lo) / (hi - lo) * 100.0
    return (tick - (lo + hi) / 2) * 360 / RESOLUTION_MINUS_1


@dataclasses.dataclass
class SafetyEnvelope:
    """Absolute per-joint bounds, in the units the policy emits.

    The calibrated range is *narrower than the servo range* — it is where the operator
    swept each joint during calibration, i.e. where the arm can actually go without
    meeting a hard stop. A policy is free to emit an action outside it, so this is the
    difference between clipping a command and driving into a mechanical limit.
    """

    limits: dict[str, tuple[float, float]]
    clipped: dict[str, int]

    @classmethod
    def from_calibration(cls, calibration, arm_margin_deg: float, gripper_margin: float):
        limits = {}
        for joint in JOINTS:
            lo = _normalized_from_tick(calibration, joint, calibration[joint].range_min)
            hi = _normalized_from_tick(calibration, joint, calibration[joint].range_max)
            margin = gripper_margin if joint == GRIPPER else arm_margin_deg
            if hi - lo <= 2 * margin:
                raise Abort(
                    f"{joint}: calibrated range {lo:.1f}..{hi:.1f} is narrower than twice "
                    f"the {margin} margin — check the calibration"
                )
            limits[joint] = (lo + margin, hi - margin)
        return cls(limits=limits, clipped={j: 0 for j in JOINTS})

    def clamp(self, joint: str, value: float) -> float:
        lo, hi = self.limits[joint]
        if value < lo or value > hi:
            self.clipped[joint] += 1
            return max(lo, min(hi, value))
        return value

    def outside(self, pose: dict[str, float]) -> list[str]:
        return [j for j in JOINTS if not self.limits[j][0] <= pose[j] <= self.limits[j][1]]


def sanity_check_calibration(calibration) -> list[str]:
    """Reject a calibration that cannot be the one the dataset was recorded in.

    ``wrist_roll`` is legitimately a full turn — ``SOFollower.calibrate`` hard-codes
    0..4095 for it and sweeps every other joint. So a *second* full-span joint means the
    sweep never captured that joint's limits, which is exactly what a known-bad
    calibration on this hardware looked like.
    """
    problems = []
    for joint in JOINTS:
        cal = calibration[joint]
        if cal.range_max <= cal.range_min:
            problems.append(f"{joint}: range_max {cal.range_max} <= range_min {cal.range_min}")
        elif joint != "wrist_roll" and (cal.range_min, cal.range_max) == (0, RESOLUTION_MINUS_1):
            problems.append(
                f"{joint}: a full 0..{RESOLUTION_MINUS_1} range means its limits were "
                f"never captured during calibration"
            )
    return problems


# ---------------------------------------------------------------------------
# The arm
# ---------------------------------------------------------------------------


class ArmDriver:
    """Bounded, releasable control of the follower arm.

    The safety contract, each item of which exists because of an observed failure:

    * **Torque is never enabled onto a stale goal.** LeRobot's ``connect()`` runs
      ``configure()``, which re-enables torque on the way out of its
      ``torque_disabled()`` block — onto whatever ``Goal_Position`` was last written,
      possibly by a previous run, possibly from before someone moved the arm by hand. So
      the steps are re-ordered here: open the bus, disable torque, write goal := present,
      *then* configure. The arm holds where it is instead of snapping to where it was.
    * **Acceleration and speed are capped**, so a bad target is slow rather than a snap.
      LeRobot's ``configure_motors()`` sets acceleration to 254 for teleoperation
      responsiveness; a policy under evaluation does not need that.
    * **Every commanded value is clamped** to the calibrated range less a margin, and to
      a per-step delta from where the arm actually is — the latter by LeRobot's
      ``max_relative_target``, which re-reads the present position on every send.
    * **Release happens on every exit path** — return to the pose the episode started
      from, then drop torque — and it does not go through any of LeRobot's
      ``@check_if_not_connected`` methods, because a camera that dropped out would
      otherwise make the release itself raise and leave the arm energised.

    The one path this cannot cover is ``SIGKILL``: an OOM kill or a lost node leaves the
    servos holding their last goal, because a Feetech bus has no watchdog. That is what
    the memory headroom, the torque limits ``configure()`` sets on the gripper, and a
    reachable physical e-stop are for.
    """

    def __init__(self, robot, envelope: SafetyEnvelope, *, no_motion: bool):
        self.robot = robot
        self.bus = robot.bus
        self.envelope = envelope
        self.no_motion = no_motion
        self.engaged = False
        self.last_commanded: dict[str, float] | None = None

    def verify_units(self) -> dict:
        """Check our normalization against LeRobot's, on the live arm, before torque.

        The guard for the one bug in this file that would otherwise be silent: a units
        mismatch drives the arm confidently to the wrong pose, with a workflow that
        reports success.
        """
        raw = self.bus.sync_read("Present_Position", normalize=False)
        norm = self.bus.sync_read("Present_Position", normalize=True)
        report, worst = {}, 0.0
        for joint in JOINTS:
            ours = _normalized_from_tick(self.bus.calibration, joint, raw[joint])
            delta = abs(ours - norm[joint])
            worst = max(worst, delta)
            report[joint] = {
                "tick": raw[joint],
                "lerobot": round(norm[joint], 4),
                "ours": round(ours, 4),
                "delta": round(delta, 5),
            }
        if worst >= 0.01:
            raise Abort(
                f"unit conversion disagrees with LeRobot by up to {worst:.4f} — refusing "
                f"to drive the arm: {json.dumps(report)}"
            )
        return report

    def present(self) -> dict[str, float]:
        return self.bus.sync_read("Present_Position")

    def observe(self) -> dict:
        """Joint positions plus one frame per camera, in the units the policy speaks."""
        return self.robot.get_observation()

    def engage(self, acceleration: int, speed: int) -> None:
        """Enable torque holding the present pose. Nothing moves."""
        if self.no_motion:
            self.last_commanded = self.present()
            print("no-motion: torque NOT enabled, no action will be sent", flush=True)
            return

        # Torque off first: a previous run may have left it on, in which case the arm is
        # holding a goal and "present" would be that goal rather than a resting pose.
        self.bus.disable_torque()
        present = self.present()
        self.bus.sync_write("Goal_Position", present)

        # Re-enables torque on the way out, now onto a goal equal to the present pose.
        self.robot.configure()
        for joint in JOINTS:
            self.bus.write("Acceleration", joint, acceleration)
            self.bus.write("Goal_Velocity", joint, speed)

        self.engaged = True
        self.last_commanded = present
        print(
            f"torque on, holding the present pose (acceleration={acceleration} speed={speed})",
            flush=True,
        )

    def send(self, pose: dict[str, float]) -> dict[str, float]:
        """Clamp to the envelope and command. Returns what was actually sent."""
        clamped = {joint: self.envelope.clamp(joint, pose[joint]) for joint in JOINTS}
        self.last_commanded = clamped
        if self.no_motion:
            return clamped
        sent = self.robot.send_action({f"{j}.pos": v for j, v in clamped.items()})
        return {k.removesuffix(".pos"): v for k, v in sent.items()}

    def ramp_to(self, target: dict[str, float], seconds: float, hz: float = 30.0) -> None:
        """Interpolate to a pose over `seconds`, so a long move is not a lunge."""
        if self.no_motion or not self.engaged:
            return
        start = self.present()
        steps = max(1, int(seconds * hz))
        for i in range(1, steps + 1):
            frac = i / steps
            self.bus.sync_write(
                "Goal_Position",
                {
                    joint: self.envelope.clamp(
                        joint, start[joint] + frac * (target[joint] - start[joint])
                    )
                    for joint in JOINTS
                },
            )
            time.sleep(1.0 / hz)
        self.last_commanded = target

    def release(self, home: dict[str, float] | None, seconds: float) -> None:
        """Return to `home` and drop torque. Must not raise, and must not be skipped.

        Deliberately built out of raw bus calls. LeRobot's ``disconnect()`` is gated on
        ``is_connected``, which includes the cameras, so a camera that failed would make
        the release path itself raise — leaving the arm holding a pose, which is the
        failure this whole file is arranged around.
        """
        if not self.engaged:
            return
        try:
            if home is not None:
                print("returning to the pose the episode started from...", flush=True)
                self.ramp_to(home, seconds)
        except Exception as exc:  # noqa: BLE001 - dropping torque matters more than this
            print(
                f"WARNING: return-to-start failed ({exc!r}); dropping torque now",
                file=sys.stderr,
                flush=True,
            )
        finally:
            try:
                self.bus.disable_torque()
                self.engaged = False
                print("torque off", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: could not disable torque: {exc!r}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


def _add_extra_dim(obs):
    """Add one leading dimension to every leaf. Called twice, for (batch=1, time=1)."""
    out = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            out[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            out[key] = _add_extra_dim(val)
        else:
            out[key] = [val]
    return out


class PolicyLink:
    """The policy server, with a timeout that suits an arm rather than a demo.

    Upstream's ``PolicyClient`` defaults to a 15 s receive timeout. At a measured ~215 ms
    round trip that is not a timeout, it is an outage: a network stall would leave the
    arm holding a pose for fifteen seconds before anything reacted. So there are two
    timeouts here, set for different jobs:

    * **setup** — generously long. The first inference after a checkpoint loads pays CUDA
      and kernel warm-up and can take tens of seconds. This phase runs with the arm limp,
      so a long wait costs nothing.
    * **the control loop** — a few multiples of the measured tail, in the hundreds of
      milliseconds. Long enough that the network's own p99 does not trip it, short enough
      that a stall is noticed inside one chunk's worth of motion.

    On a loop timeout the arm **holds its last commanded pose** and the request is
    retried. Holding is the right immediate answer to one dropped chunk: the arm is at a
    pose the policy chose, and it is not moving. Going limp on a single lost packet is
    worse — the arm falls under gravity and drops whatever it is holding. But holding is
    only correct *briefly*, so when the retries are exhausted the episode aborts and the
    arm is returned and released. The timeout behaviour is therefore: hold, retry, then
    release.
    """

    def __init__(self, host, port, *, loop_timeout_ms, setup_timeout_s, retries):
        from gr00t.policy.server_client import PolicyClient

        self._client_cls = PolicyClient
        self.host, self.port = host, port
        self.loop_timeout_ms = loop_timeout_ms
        self.setup_timeout_ms = int(setup_timeout_s * 1000)
        self.retries = retries
        self.client = None
        self.active_timeout_ms = loop_timeout_ms
        self.latencies_ms: list[float] = []
        self.timeouts = 0

    def _open(self, timeout_ms: int) -> None:
        if self.client is not None:
            self.client.close()
        # timeout_ms is applied when the socket is created, so switching phases means a
        # new client rather than mutating this one.
        self.client = self._client_cls(host=self.host, port=self.port, timeout_ms=timeout_ms)
        self.active_timeout_ms = timeout_ms

    def open_for_setup(self) -> None:
        self._open(self.setup_timeout_ms)

    def open_for_loop(self) -> None:
        self._open(self.loop_timeout_ms)

    def wait_until_ready(self, deadline_s: float, poll_timeout_ms: int = 5000) -> float:
        """Poll until the server answers a ping, or give up at `deadline_s`.

        The group is gang-scheduled, so this task starts at the same moment the policy
        task does — and that task has to download a ~24 Gi checkpoint, build its
        toolchain and load a 3B model first. A single ping would fail long before any of
        that finished, so the wait is explicit and bounded rather than a fixed sleep.
        """
        self._open(poll_timeout_ms)
        start = time.perf_counter()
        attempts = 0
        while True:
            if _stop.is_set():
                raise Abort("stop requested while waiting for the policy server")
            if self.client.ping():
                waited = time.perf_counter() - start
                print(
                    f"policy server answered a ping after {waited:.0f}s "
                    f"({attempts + 1} attempts)",
                    flush=True,
                )
                return waited
            attempts += 1
            waited = time.perf_counter() - start
            if waited >= deadline_s:
                raise Abort(
                    f"policy server at {self.host}:{self.port} did not answer within "
                    f"{deadline_s:.0f}s — check the policy-server task's own logs, it is "
                    f"probably still downloading the checkpoint or loading the model"
                )
            if attempts % 6 == 1:
                print(f"  still waiting for the policy server ({waited:.0f}s)", flush=True)

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    @staticmethod
    def build_observation(obs: dict, instruction: str) -> dict:
        """Upstream's observation contract, unchanged.

        ``float32`` is not incidental: the server validates dtypes and rejects anything
        else, which is the good outcome — the alternative is a silently rescaled state.
        """
        state = np.array([obs[key] for key in STATE_KEYS], dtype=np.float32)
        model_obs = {
            "video": {"front": obs["front"], "wrist": obs["wrist"]},
            "state": {"single_arm": state[:5], "gripper": state[5:6]},
            "language": {LANGUAGE_KEY: instruction},
        }
        return _add_extra_dim(_add_extra_dim(model_obs))

    def payload_bytes(self, obs: dict, instruction: str) -> int:
        """Size of one observation on the wire.

        Worth reporting rather than assuming. The frames are sent as *raw* uint8 arrays,
        so two 640x480 cameras are ~1.8 MB per request — orders of magnitude more than a
        compressed frame, and on a lab uplink that transfer, not inference, is what
        dominates the round trip.
        """
        from gr00t.policy.server_client import MsgSerializer

        request = {
            "endpoint": "get_action",
            "data": {"observation": self.build_observation(obs, instruction), "options": None},
        }
        return len(MsgSerializer.to_bytes(request))

    def request(self, obs: dict, instruction: str) -> list[dict[str, float]]:
        """One planning request. Returns the chunk as a list of per-step poses."""
        model_obs = self.build_observation(obs, instruction)
        last_error = None
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                chunk, _info = self.client.get_action(model_obs)
            except zmq.error.Again as exc:
                # The client has already rebuilt its socket, so the next attempt can
                # send. The arm is still holding its last commanded pose.
                self.timeouts += 1
                last_error = exc
                print(
                    f"WARNING: policy request timed out after {self.active_timeout_ms} ms "
                    f"(attempt {attempt + 1}/{self.retries + 1}); holding pose",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            self.latencies_ms.append((time.perf_counter() - start) * 1000.0)
            return self._decode(chunk)
        raise Abort(
            f"policy server did not answer within {self.active_timeout_ms} ms across "
            f"{self.retries + 1} attempts ({last_error!r})"
        )

    @staticmethod
    def _decode(chunk: dict) -> list[dict[str, float]]:
        arm = np.asarray(chunk["single_arm"])[0]  # (T, 5)
        grip = np.asarray(chunk["gripper"])[0]  # (T, 1)
        return [
            dict(
                zip(JOINTS, [float(v) for v in arm[t]] + [float(grip[t][0])], strict=True)
            )
            for t in range(arm.shape[0])
        ]


# ---------------------------------------------------------------------------
# Operator control
# ---------------------------------------------------------------------------


class ControlDir:
    """Sentinel files a human can touch, since a container has no keyboard.

    The simulation harness this loop is modelled on resets an episode on a keypress. On
    real hardware the equivalent has to be reachable from outside the container, and it
    is: ``osmo workflow exec`` rides the same API-server-to-kubelet path the logs do.
    """

    NAMES = ("abort", "success", "fail")

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        for name in self.NAMES:
            (self.path / name).unlink(missing_ok=True)

    def take(self) -> str | None:
        for name in self.NAMES:
            marker = self.path / name
            if marker.exists():
                marker.unlink(missing_ok=True)
                return name
        return None


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


def run_episode(index, arm, policy, control, rate, args, trace, records) -> None:
    """Run one episode, append its record, and never leave the arm moving.

    The record is appended in a ``finally`` rather than returned, so an abort halfway
    through an episode still publishes what happened up to that point — which is the
    interesting part when something went wrong.
    """
    start_pose = arm.present()
    state = {"outcome": "completed", "success": None, "note": "", "steps": 0, "replans": 0}
    # Commanded-versus-achieved, sampled once per chunk. This arm does not reach what it
    # is told — unevenly, per joint — so the gap is a first-class result, not a detail.
    achieved_gap: dict[str, list[float]] = {j: [] for j in JOINTS}
    episode_start = time.perf_counter()

    try:
        if args.home_pose:
            print(f"episode {index}: ramping to the home pose over {args.home_ramp_s}s", flush=True)
            arm.ramp_to(args.home_pose, args.home_ramp_s)

        rate.reset()
        while True:
            if _stop.is_set():
                raise Abort("stop requested")
            marker = control.take()
            if marker == "abort":
                raise Abort("operator touched the abort file")
            if marker in ("success", "fail"):
                state["outcome"] = "scored"
                state["success"] = marker == "success"
                break
            if (time.perf_counter() - episode_start) >= args.episode_timeout_s:
                state["outcome"] = "timeout"
                break

            obs = arm.observe()
            chunk = policy.request(obs, args.lang_instruction)
            state["replans"] += 1
            if args.action_horizon > len(chunk):
                # Upstream raises here too, and it is right to: executing a shorter chunk
                # than asked for silently changes the control rate the latency budget was
                # built on.
                raise Abort(
                    f"--action-horizon {args.action_horizon} exceeds the policy's chunk "
                    f"length {len(chunk)}"
                )

            last_sent = None
            for target in chunk[: args.action_horizon]:
                last_sent = arm.send(target)
                state["steps"] += 1
                if trace is not None:
                    trace.write(
                        json.dumps(
                            {
                                "episode": index,
                                "step": state["steps"],
                                "t": round(time.perf_counter() - episode_start, 4),
                                "requested": {j: round(v, 3) for j, v in target.items()},
                                "sent": {j: round(v, 3) for j, v in last_sent.items()},
                            }
                        )
                        + "\n"
                    )
                rate.sleep()
                if _stop.is_set():
                    raise Abort("stop requested")
                if (time.perf_counter() - episode_start) >= args.episode_timeout_s:
                    state["outcome"] = "timeout"
                    break

            if last_sent is not None and not arm.no_motion:
                reached = arm.present()
                for joint in JOINTS:
                    achieved_gap[joint].append(reached[joint] - last_sent[joint])
            if state["outcome"] == "timeout":
                break
    except Abort as exc:
        state["outcome"] = "aborted"
        state["note"] = str(exc)
        raise
    finally:
        duration = time.perf_counter() - episode_start
        records.append(
            {
                "episode": index,
                "outcome": state["outcome"],
                "success": state["success"],
                "note": state["note"],
                "duration_s": round(duration, 2),
                "steps": state["steps"],
                "replans": state["replans"],
                "achieved_hz": round(state["steps"] / duration, 2) if duration > 0 else None,
                "start_pose": {j: round(v, 3) for j, v in start_pose.items()},
                "mean_shortfall": {
                    j: round(statistics.fmean(v), 3) for j, v in achieved_gap.items() if v
                },
            }
        )
        # Return to where this episode began, then release, whatever happened.
        arm.release(start_pose, args.release_ramp_s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _percentiles(samples: list[float]) -> dict:
    if not samples:
        return {}
    array = np.asarray(samples)
    return {
        "n": len(samples),
        "min_ms": round(float(array.min()), 2),
        "p50_ms": round(float(np.percentile(array, 50)), 2),
        "p95_ms": round(float(np.percentile(array, 95)), 2),
        "p99_ms": round(float(np.percentile(array, 99)), 2),
        "max_ms": round(float(array.max()), 2),
    }


def _parse_pose(text: str | None) -> dict[str, float] | None:
    if not text:
        return None
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != len(JOINTS):
        raise SystemExit(f"--home-pose needs {len(JOINTS)} values in the order {JOINTS}")
    return dict(zip(JOINTS, values, strict=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    g = p.add_argument_group("policy server")
    g.add_argument("--policy-host", required=True)
    g.add_argument("--policy-port", type=int, default=5555)
    g.add_argument(
        "--policy-timeout-ms",
        type=int,
        default=1500,
        help="Receive timeout during the control loop. Not upstream's 15000: with "
        "--policy-retries this bounds how long the arm can hold a stale pose at "
        "(retries + 1) x timeout. Must exceed the round-trip p99 for your payload.",
    )
    g.add_argument(
        "--policy-setup-timeout-s",
        type=float,
        default=300.0,
        help="Receive timeout for the ping and warm-up calls, which run with the arm "
        "limp and include the model's first-inference cost.",
    )
    g.add_argument("--policy-retries", type=int, default=1)
    g.add_argument(
        "--policy-wait-s",
        type=float,
        default=1800.0,
        help="How long to keep pinging before giving up on the policy server. The group "
        "is gang-scheduled, so the server is downloading its checkpoint and loading the "
        "model while this task is already running.",
    )
    g.add_argument("--warmup-requests", type=int, default=3)

    g = p.add_argument_group("robot")
    g.add_argument("--robot-port", default="/dev/ttyACM0")
    g.add_argument("--robot-id", default="follower")
    g.add_argument(
        "--calibration-dir",
        default=None,
        help="Directory holding <robot-id>.json. When given, the servos' own EEPROM is "
        "asserted equal to it — the strongest available check that the arm is in the "
        "joint frame the dataset was recorded in. When omitted, the calibration is read "
        "from the servos, which is where LeRobot stores it.",
    )
    g.add_argument("--front-camera", default="/dev/video2")
    g.add_argument("--wrist-camera", default="/dev/video4")
    g.add_argument("--camera-width", type=int, default=640)
    g.add_argument("--camera-height", type=int, default=480)
    g.add_argument("--camera-fps", type=int, default=30)

    g = p.add_argument_group("safety")
    g.add_argument(
        "--joint-margin-deg",
        type=float,
        default=5.0,
        help="Shrink each arm joint's calibrated range by this much at both ends.",
    )
    g.add_argument("--gripper-margin", type=float, default=3.0)
    g.add_argument(
        "--max-step-deg",
        type=float,
        default=8.0,
        help="Per-step cap on how far an arm joint may be commanded from where it "
        "actually is. At 30 Hz the default is far above the arm's real speed, so it "
        "bounds a wild action without limiting normal motion.",
    )
    g.add_argument("--max-step-gripper", type=float, default=15.0)
    g.add_argument("--acceleration", type=int, default=20, help="Servo Acceleration register.")
    g.add_argument("--speed", type=int, default=600, help="Servo Goal_Velocity register.")
    g.add_argument(
        "--no-motion",
        action="store_true",
        help="Dress rehearsal: connect, warm up, run the whole loop and log every action "
        "that would be sent — but never enable torque. The arm stays limp.",
    )
    g.add_argument("--release-ramp-s", type=float, default=2.0)

    g = p.add_argument_group("episodes")
    g.add_argument("--lang-instruction", required=True)
    g.add_argument("--action-horizon", type=int, default=8)
    g.add_argument("--control-hz", type=float, default=30.0)
    g.add_argument("--max-episodes", type=int, default=1)
    g.add_argument("--episode-timeout-s", type=float, default=60.0)
    g.add_argument(
        "--home-pose",
        default=None,
        help="Six values in JOINTS order. When given, each episode ramps here first, "
        "which is what makes repeated episodes comparable.",
    )
    g.add_argument("--home-ramp-s", type=float, default=4.0)

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", required=True)
    g.add_argument("--control-dir", default="/tmp/hil-control")  # noqa: S108 - task-local
    return p


def main() -> int:  # noqa: PLR0915 - a linear bring-up sequence, ordered by risk
    args = build_parser().parse_args()
    args.home_pose = _parse_pose(args.home_pose)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.motors import MotorCalibration
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    _install_signal_handlers()
    control = ControlDir(pathlib.Path(args.control_dir))
    print(
        "operator control: touch "
        + " or ".join(f"{args.control_dir}/{n}" for n in ControlDir.NAMES)
        + "  (reachable with `osmo workflow exec`)",
        flush=True,
    )

    records: list[dict] = []
    results = {
        "config": {k: v for k, v in vars(args).items()},
        "episodes": records,
    }
    arm = None
    envelope = None
    rate = RateLimiter(args.control_hz)
    cameras_connected = []

    policy = PolicyLink(
        args.policy_host,
        args.policy_port,
        loop_timeout_ms=args.policy_timeout_ms,
        setup_timeout_s=args.policy_setup_timeout_s,
        retries=args.policy_retries,
    )

    try:
        # ---- 1. the policy server, before anything touches the robot ---------
        # Deliberately first. It proves the group scheduled, that the task host token
        # resolved, and that the port is open — all checkable with the arm powered off,
        # which is the last rehearsal before one that can move.
        print(f"connecting to the policy server at {args.policy_host}:{args.policy_port}", flush=True)
        results["policy_wait_s"] = round(policy.wait_until_ready(args.policy_wait_s), 1)
        policy.open_for_setup()

        # ---- 2. the robot bus ------------------------------------------------
        calibration = None
        if args.calibration_dir:
            path = pathlib.Path(args.calibration_dir) / f"{args.robot_id}.json"
            raw = json.loads(path.read_text())
            calibration = {j: MotorCalibration(**raw[j]) for j in JOINTS}
            print(f"calibration file: {path}", flush=True)

        robot = SO101Follower(
            SO101FollowerConfig(
                port=args.robot_port,
                id=args.robot_id,
                calibration_dir=(
                    pathlib.Path(args.calibration_dir) if args.calibration_dir else None
                ),
                # Applied inside send_action against the present position, so it caps how
                # fast a joint can be driven regardless of what the policy asks for.
                max_relative_target={
                    **{j: args.max_step_deg for j in ARM_JOINTS},
                    GRIPPER: args.max_step_gripper,
                },
                cameras={
                    name: OpenCVCameraConfig(
                        index_or_path=pathlib.Path(device),
                        fps=args.camera_fps,
                        width=args.camera_width,
                        height=args.camera_height,
                    )
                    for name, device in (
                        ("front", args.front_camera),
                        ("wrist", args.wrist_camera),
                    )
                },
            )
        )

        # Not robot.connect(): that runs configure(), which enables torque, before any of
        # the checks below have had a chance to refuse.
        robot.bus.connect()
        print(f"bus open on {args.robot_port}, six motors answered", flush=True)

        if calibration is not None:
            if not robot.bus.is_calibrated:
                raise Abort(
                    "the servos' EEPROM does not match the calibration file, so the arm "
                    "is not in the joint frame the dataset was recorded in — refusing to "
                    "drive it"
                )
            print("EEPROM matches the calibration file", flush=True)
        else:
            calibration = robot.bus.read_calibration()
            robot.bus.calibration = calibration
            print("calibration read from the servos", flush=True)

        problems = sanity_check_calibration(calibration)
        if problems:
            raise Abort("calibration is not usable: " + "; ".join(problems))
        results["calibration"] = {j: dataclasses.asdict(c) for j, c in calibration.items()}

        envelope = SafetyEnvelope.from_calibration(
            calibration, args.joint_margin_deg, args.gripper_margin
        )
        results["envelope"] = {j: [round(v, 2) for v in lim] for j, lim in envelope.limits.items()}
        print("safety envelope:", flush=True)
        for joint, (lo, hi) in envelope.limits.items():
            print(f"  {joint:<14} {lo:>8.2f} .. {hi:>8.2f}", flush=True)

        arm = ArmDriver(robot, envelope, no_motion=args.no_motion)
        results["units_check"] = arm.verify_units()
        print("unit conversion agrees with LeRobot on all six joints", flush=True)

        outside = envelope.outside(arm.present())
        results["resting_pose_outside_envelope"] = outside
        if outside:
            # Not fatal — the first policy action is simply clamped — but it means the arm
            # is resting past where it was calibrated to sweep, and the operator should
            # know that before torque goes on.
            print(
                f"WARNING: the resting pose is outside the safety envelope on {outside}; "
                f"the first commanded action will be clamped",
                file=sys.stderr,
                flush=True,
            )

        # ---- 3. the cameras ---------------------------------------------------
        for name, cam in robot.cameras.items():
            cam.connect()
            cameras_connected.append(cam)
            print(f"camera {name} open", flush=True)

        # ---- 4. warm-up, with the arm still limp ------------------------------
        # The whole policy path — observation shape, dtypes, server-side validation, chunk
        # length, and whether the returned actions are even inside the arm's reachable
        # range — is exercised here, before torque. If any of it is wrong, the run ends
        # with the arm untouched.
        obs = robot.get_observation()
        payload = policy.payload_bytes(obs, args.lang_instruction)
        results["observation_bytes"] = payload
        print(
            f"observation payload: {payload / 1e6:.2f} MB per request "
            f"({args.camera_width}x{args.camera_height} raw uint8 x2)",
            flush=True,
        )
        warmup_ms, chunk = [], None
        for i in range(max(1, args.warmup_requests)):
            start = time.perf_counter()
            chunk = policy.request(obs, args.lang_instruction)
            warmup_ms.append(round((time.perf_counter() - start) * 1000.0, 1))
            print(
                f"warm-up request {i + 1}: {warmup_ms[-1]} ms, chunk {len(chunk)} steps",
                flush=True,
            )
        policy.latencies_ms.clear()  # warm-up is not steady state
        results["warmup_ms"] = warmup_ms
        results["chunk_length"] = len(chunk)

        # Check the loop can actually close, and say so before torque rather than leaving
        # it to be inferred from a stuttering arm. One chunk buys action_horizon/control_hz
        # of motion; if a round trip costs more than that, the arm reaches the end of every
        # chunk before the next one arrives and the achieved rate falls below control_hz.
        budget_ms = args.action_horizon / args.control_hz * 1000.0
        steady_ms = min(warmup_ms[1:]) if len(warmup_ms) > 1 else warmup_ms[0]
        results["budget_ms"] = round(budget_ms, 1)
        results["fits_budget"] = steady_ms < budget_ms
        print(
            f"budget: horizon {args.action_horizon} at {args.control_hz:g} Hz buys "
            f"{budget_ms:.0f} ms of motion; a round trip costs about {steady_ms:.0f} ms",
            flush=True,
        )
        if steady_ms >= budget_ms:
            needed = int(np.ceil(steady_ms * args.control_hz / 1000.0))
            print(
                f"WARNING: the control loop will not hold {args.control_hz:g} Hz. A round "
                f"trip costs more than one chunk of motion, so the arm will pause at every "
                f"chunk boundary.\n"
                f"         The payload is {payload / 1e6:.2f} MB of raw frames, which on a "
                f"lab uplink is most of that cost — so the levers, in order of effect, are: "
                f"reduce --camera-width/--camera-height, raise --action-horizon to at least "
                f"{needed} (ceiling {len(chunk)}), or move the policy closer.",
                file=sys.stderr,
                flush=True,
            )

        if args.action_horizon > len(chunk):
            raise Abort(
                f"--action-horizon {args.action_horizon} exceeds the policy's chunk length "
                f"{len(chunk)}; the server cannot supply that many steps"
            )
        first_outside = sorted({j for step in chunk for j in envelope.outside(step)})
        results["warmup_actions_outside_envelope"] = first_outside
        if first_outside:
            print(
                f"WARNING: the first chunk asks for poses outside the envelope on "
                f"{first_outside}; those commands will be clamped",
                file=sys.stderr,
                flush=True,
            )

        # ---- 5. episodes -------------------------------------------------------
        policy.close()
        policy.open_for_loop()
        print(f"switching to the control-loop timeout of {args.policy_timeout_ms} ms", flush=True)

        with (output / "trace.jsonl").open("w") as trace:
            for index in range(1, args.max_episodes + 1):
                arm.engage(args.acceleration, args.speed)
                run_episode(index, arm, policy, control, rate, args, trace, records)
                print(
                    f"episode {index}: {records[-1]['outcome']} "
                    f"({records[-1]['steps']} steps at {records[-1]['achieved_hz']} Hz)",
                    flush=True,
                )

    except Abort as exc:
        results["aborted"] = str(exc)
        print(f"\nABORTED: {exc}", file=sys.stderr, flush=True)
    except Exception as exc:  # noqa: BLE001 - record it, release the arm, then re-raise
        results["error"] = f"{type(exc).__name__}: {exc}"
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        # The one thing that must always happen.
        if arm is not None:
            arm.release(arm.last_commanded, args.release_ramp_s)
        for cam in cameras_connected:
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001,S110 - teardown, nothing left to salvage
                pass
        policy.close()

        scored = [e for e in records if e.get("success") is not None]
        successes = sum(1 for e in scored if e["success"])
        results["summary"] = {
            "episodes_run": len(records),
            "episodes_scored": len(scored),
            "successes": successes,
            "success_rate": round(successes / len(scored), 3) if scored else None,
            "chunk_latency": _percentiles(policy.latencies_ms),
            "policy_timeouts": policy.timeouts,
            "loop_overruns": rate.overruns,
            "worst_loop_overrun_ms": round(rate.worst_overrun_ms, 1),
            "clipped_by_joint": dict(envelope.clipped) if envelope else {},
            "no_motion": args.no_motion,
        }
        (output / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
        print("\n" + json.dumps(results["summary"], indent=2), flush=True)

    return 1 if results.get("aborted") else 0


if __name__ == "__main__":
    sys.exit(main())
