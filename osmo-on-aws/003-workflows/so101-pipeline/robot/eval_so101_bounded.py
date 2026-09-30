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

Units. This file has no notion of servo ticks. The environment reports its joint limits in the
same space the policy emits actions in, and does whatever conversion that needs internally — so
the clamp and the action it bounds are in one space by construction, rather than by a conversion
restated here and checked against the device.
"""

# Required, not cosmetic: `Environment` below is imported only for type checkers, so without
# postponed evaluation the annotations referencing it raise NameError at import.
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import signal
import statistics
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import zmq

if TYPE_CHECKING:
    # Names the object `ArmDriver` drives. Guarded because `environment.py` is injected beside this
    # file rather than installed; the implementation chosen for the run is imported inside main().
    from environment import Environment

# Joint order is load-bearing: it is the order the dataset was recorded in, the order
# the checkpoint's action head emits, and the order the servos are addressed in.
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]
GRIPPER = "gripper"
STATE_KEYS = [f"{j}.pos" for j in JOINTS]

LANGUAGE_KEY = "annotation.human.task_description"


# Where the ZMQ transport comes from. Default is the reference implementation this client
# was built against; a second model can supply its own without a second copy of this file,
# which matters because everything below `ArmDriver` is hardware-safety code earned on real
# hardware and must not exist twice. The transport itself is model-agnostic — REQ/REP plus
# msgpack-numpy with a pickle-safety boundary — so only the import location varies.
TRANSPORT_MODULE = "gr00t.policy.server_client"


def _transport():
    """Import the transport module named by --transport-module.

    Resolved lazily and through one function, so the client class and the serializer cannot
    end up coming from different implementations and disagreeing about the wire format.
    """
    import importlib

    return importlib.import_module(TRANSPORT_MODULE)


class Abort(Exception):
    """Raised to unwind through the release path rather than dying where we stand."""


class Cooldown(Exception):
    """Raised to end an interval early so the servos can rest, without ending the run.

    Distinct from ``Abort`` because it is not a fault: it unwinds the same way, returning
    the arm and dropping torque, but the caller then waits for the servos to cool and
    starts a fresh interval instead of giving up.
    """


_stop = threading.Event()


# ---------------------------------------------------------------------------
# Cameras
#
# Only the NAME is this file's business. A name is the dataset's image-feature suffix and
# goes on the wire to the policy server, which matches on it. What sits behind the name is
# the environment's business: resolving a spec to a device is Linux-specific and lives in
# the implementation, which reports what it selected through `camera_sources()`.
# ---------------------------------------------------------------------------


def _parse_cameras(values: list[str]) -> dict[str, str]:
    """``NAME=SPEC`` pairs into an ordered mapping, with the wire-name rule enforced."""
    cameras: dict[str, str] = {}
    for value in values or []:
        name, sep, spec = value.partition("=")
        if not sep or not name or not spec:
            raise Abort(f"--camera wants NAME=SPEC, got {value!r}")
        # The name goes on the wire and into a dataset feature key, where a dot or a comma
        # changes how something downstream parses it. Same rule the pipeline's preflight applies.
        bad = [c for c in ".,= \t" if c in name]
        if bad:
            raise Abort(f"camera name {name!r} must not contain {''.join(bad)!r}")
        if name in cameras:
            raise Abort(f"camera {name!r} given twice")
        cameras[name] = spec
    if not cameras:
        raise Abort("at least one --camera is required")
    return cameras


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


@dataclasses.dataclass
class SafetyEnvelope:
    """Absolute per-joint bounds, in the units the policy emits.

    Built from the travel the environment reports, less a margin. Those limits are narrower than
    whatever the device is mechanically capable of — on a real arm they are where the operator
    swept each joint, i.e. where it can go without meeting a hard stop. A policy is free to emit
    an action outside them, so this is the difference between clipping a command and driving into
    a limit.
    """

    limits: dict[str, tuple[float, float]]
    clipped: dict[str, int]

    @classmethod
    def from_limits(cls, limits: dict[str, tuple[float, float]], arm_margin_deg: float,
                    gripper_margin: float):
        """Shrink each joint's reported travel by a margin, in the units actions arrive in.

        No conversion happens here. The environment reports its limits in the same space the
        policy emits, so this only steps back from the ends.
        """
        bounds = {}
        for joint in JOINTS:
            lo, hi = limits[joint]
            margin = gripper_margin if joint == GRIPPER else arm_margin_deg
            lo, hi = lo + margin, hi - margin
            if lo >= hi:
                raise Abort(
                    f"{joint}: travel {limits[joint]} is narrower than twice "
                    f"the {margin} margin — check the environment's limits"
                )
            bounds[joint] = (lo, hi)
        return cls(limits=bounds, clipped={j: 0 for j in JOINTS})

    def clamp(self, joint: str, value: float) -> float:
        lo, hi = self.limits[joint]
        if value < lo or value > hi:
            self.clipped[joint] += 1
            return max(lo, min(hi, value))
        return value

    def outside(self, pose: dict[str, float]) -> list[str]:
        return [j for j in JOINTS if not self.limits[j][0] <= pose[j] <= self.limits[j][1]]


def sanity_check_limits(limits: dict[str, tuple[float, float]]) -> list[str]:
    """Reject limits no environment should report, whatever it is.

    Only the unit-agnostic check lives here: a range must be a range. Whether a range is
    *plausible* for a given device is the environment's own business, because only it knows what
    its own uncalibrated state looks like, and it refuses rather than reporting one.
    """
    problems = []
    for joint in JOINTS:
        if joint not in limits:
            problems.append(f"{joint}: no limits reported")
            continue
        lo, hi = limits[joint]
        if hi <= lo:
            problems.append(f"{joint}: upper limit {hi} <= lower limit {lo}")
    return problems


def _torque_is_off(state: dict[str, int] | None) -> bool:
    """Is torque *known* to be off, from a reading rather than from having asked?

    Three inputs, three meanings, and the middle one is why this is a function rather than an
    inline `all()`:

    * ``None`` — this environment has no notion of torque. Nothing is holding, by construction.
    * ``{}`` — the state could not be read. **Not** released: `all()` over an empty dict is
      vacuously true, which would turn the one reading that means "the bus went quiet" into
      proof of the thing it failed to measure.
    * a populated dict — released only if every joint reads zero.
    """
    if state is None:
        return True
    return bool(state) and all(value == 0 for value in state.values())


def force_torque_off(env: Environment, attempts: int = 4) -> dict[str, int] | None:
    """Drop torque and **verify** it, retrying. Never raises. Returns the last state read.

    Verified rather than assumed, because this is the one operation the whole file exists to
    guarantee. Two things make a single unverified write insufficient: a Feetech write can fail
    transiently at 1 Mbaud, and a servo that browns out during a write sequence comes back with
    its EEPROM default — which on this arm is torque *enabled*. So a release that is not read
    back is not a release.

    Takes an environment rather than living on ``ArmDriver``, because the arm can already be
    energised from an earlier run at a point where no driver exists yet — the driver needs an
    envelope, which needs limits, which need a connected environment. Releasing must not wait
    for that.
    """
    state: dict[str, int] | None = {}
    for attempt in range(attempts):
        try:
            env.disable_torque()
        except Exception as exc:  # noqa: BLE001 - keep going, then check
            print(
                f"WARNING: disable_torque attempt {attempt + 1} failed: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
        try:
            state = env.torque_state()
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: could not read back torque state: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.2)
            continue
        if _torque_is_off(state):
            return state
        still_on = [j for j, v in state.items() if v]
        if still_on:
            why = f"torque still enabled on {still_on}"
        else:
            # An empty read-back, so nothing was confirmed either way. The retry is the point:
            # this is the reading that must not be mistaken for a release.
            why = "torque state read back empty, so the release is unverified"
        remaining = attempts - attempt - 1
        print(
            f"WARNING: {why} after attempt {attempt + 1}/{attempts}"
            f"{'; retrying' if remaining else ''}",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(0.2)
    return state


def report_torque(state: dict[str, int] | None) -> None:
    """Say, in operator terms, whether the arm is limp. The last line of any run that drove."""
    if state is None:
        print("this environment has no torque to release", flush=True)
    elif _torque_is_off(state):
        print("torque off, verified on all six joints", flush=True)
    else:
        # The arm may still be holding a pose and software has run out of ways to stop it.
        # Say so in the strongest terms available.
        print(
            f"ERROR: COULD NOT RELEASE TORQUE — read back {state or 'nothing'}. "
            f"The arm may still be energised. CUT POWER TO THE ARM.",
            file=sys.stderr,
            flush=True,
        )


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
    * **Release happens on every exit path** — return to the pose the interval started
      from, then drop torque — and it does not go through any of LeRobot's
      ``@check_if_not_connected`` methods, because a camera that dropped out would
      otherwise make the release itself raise and leave the arm energised.

    The one path this cannot cover is ``SIGKILL``: an OOM kill or a lost node leaves the
    servos holding their last goal, because a Feetech bus has no watchdog. That is what
    the memory headroom, the torque limits ``configure()`` sets on the gripper, and a
    reachable physical e-stop are for.
    """

    def __init__(
        self,
        env: Environment,
        envelope: SafetyEnvelope,
        *,
        no_motion: bool,
        settle_timeout_s: float = 6.0,
        deadband_deg: float = 0.0,
        deadband_gripper: float = 0.0,
    ):
        self.env = env
        self.envelope = envelope
        self.no_motion = no_motion
        self.engaged = False
        self.last_commanded: dict[str, float] | None = None
        self.final_torque_state: dict[str, int] | None = {}
        self.settle_timeout_s = settle_timeout_s
        self.return_residual: float | None = None
        self.deadband_deg = deadband_deg
        self.deadband_gripper = deadband_gripper
        self.held = {j: 0 for j in JOINTS}
        self.bad_temp_reads = 0
        self.hot_streak = 0
        self.last_temp: dict[str, int] = {}
        self.temp_log: list[dict] = []

    def present(self) -> dict[str, float]:
        return self.env.present_position()

    def observe(self) -> dict:
        """Joint positions plus one frame per camera, in the units the policy speaks."""
        return self.env.get_observation()

    def torque_state(self) -> dict[str, int] | None:
        """Per-joint torque-enable, passed through unflattened. See `_torque_is_off`.

        None and `{}` are both falsy and mean opposite things, so every caller here goes through
        `_torque_is_off` rather than testing the dict.
        """
        return self.env.torque_state()

    # Above the servos' Max_Temperature_Limit (register 0x0D), which the vendor memory table
    # documents in degrees Celsius and which reads 70 on this arm. A value past it is a
    # corrupted packet rather than a hot motor: the bus mangles one occasionally, and a byte
    # read wrong is indistinguishable from a real value except by being impossible.
    IMPLAUSIBLE_TEMP_C = 90

    # Degrees a servo may plausibly change between consecutive reads, which are about a
    # second apart. Thermal mass makes a larger jump impossible; a wrong byte makes it easy.
    MAX_TEMP_SLEW_C = 6

    def temperatures(self) -> dict[str, int] | None:
        """Servo temperatures, with physically impossible readings dropped.

        Returns None when the environment does not model heat at all — distinct from `{}`, which
        means a reading was attempted and every value was rejected. The first means there is
        nothing to manage; the second means the bus went quiet and the arm must be assumed hot.

        Two filters, because the obvious one is not the useful one. An absolute ceiling
        catches a reading of 150 C, but every *ambiguous* reading this arm has produced sat
        between 38 and 61 C, which sails straight through it.

        The discriminating filter is **slew rate**: a servo has thermal mass and cannot move
        20 C between reads a second apart, but a corrupted byte can. So a reading is rejected
        when it implies an impossible rate of change from the last accepted one for that
        joint. That catches corruption inside the plausible band, which is where it matters —
        a spurious 55 C is what triggers a rest, not a spurious 150 C.

        Every reading is also recorded, so a run answers the question "was it ever really
        hot" from its own trace instead of from a maximum that one bad read can inflate.
        """
        raw = self.env.temperatures()
        if raw is None:
            return None
        clean, dropped = {}, {}
        for joint, value in raw.items():
            if value is None or not 0 < value < self.IMPLAUSIBLE_TEMP_C:
                dropped[joint] = value
                continue
            previous = self.last_temp.get(joint)
            if previous is not None and abs(value - previous) > self.MAX_TEMP_SLEW_C:
                dropped[joint] = value
                continue
            clean[joint] = value
            self.last_temp[joint] = value
        if dropped:
            self.bad_temp_reads += len(dropped)
            print(
                f"WARNING: rejecting temperature reading(s) {dropped} as bus corruption "
                f"(last accepted {({j: self.last_temp.get(j) for j in dropped})}); a servo "
                f"cannot change that fast, and one this hot would be past its own limit",
                file=sys.stderr,
                flush=True,
            )
        if clean:
            self.temp_log.append({"t": round(time.time(), 2), **clean})
        return clean

    def wait_to_cool(self, resume_c: int, timeout_s: float, poll_s: float = 5.0) -> dict:
        """Rest until every servo is back under `resume_c`. Torque must already be off.

        Cooling only happens with torque released — a servo holding a pose against gravity
        keeps drawing current — so the caller returns the arm and drops torque first, and
        the arm is limp for the duration of this wait.

        `resume_c` is deliberately well below the pause threshold. Resuming the moment the
        ceiling is cleared would stop and start every few seconds; the gap is the
        hysteresis that makes a rest an actual rest.
        """
        def hottest(reading: dict[str, int] | None) -> int:
            # An empty reading means every joint's value was rejected as corruption. Treat
            # that as "still hot" rather than "cool": resuming on a bus that cannot be read
            # would mean resuming with no thermal protection at all.
            return max(reading.values()) if reading else self.IMPLAUSIBLE_TEMP_C

        start = time.perf_counter()
        temps = self.temperatures()
        if temps is None:
            # Nothing to wait for, and waiting would time out and abort a healthy run. Reaching
            # here at all means something upstream asked a heatless environment to cool.
            print("this environment does not report temperature; nothing to rest for", flush=True)
            return {"rested_s": 0.0, "peak": {}, "resumed_at": {}}
        peak = dict(temps)
        print(
            f"resting until every servo is under {resume_c}C (now "
            f"{max(temps.values()) if temps else 'unreadable'}C, torque off)",
            flush=True,
        )
        while hottest(temps) >= resume_c:
            if _stop.is_set():
                raise Abort("stop requested while resting")
            waited = time.perf_counter() - start
            if waited >= timeout_s:
                raise Abort(
                    f"servos did not cool below {resume_c}C within {timeout_s:.0f}s "
                    f"(still {temps}); something is holding current or the arm is stalled"
                )
            time.sleep(poll_s)
            temps = self.temperatures()
            if int(waited) % 30 < poll_s:
                # The reading just taken, not a fresh one: an extra read would move the slew
                # filter's baseline, so a progress line would change the next real decision.
                print(f"  resting, {waited:.0f}s elapsed, hottest {hottest(temps)}C", flush=True)
        rested = time.perf_counter() - start
        print(f"cooled to {hottest(temps)}C after {rested:.0f}s; resuming", flush=True)
        self.hot_streak = 0
        return {"rested_s": round(rested, 1), "peak": peak, "resumed_at": temps}

    def force_torque_off(self, attempts: int = 4) -> dict[str, int] | None:
        """Drop torque and verify it. Never raises. See `force_torque_off` at module level."""
        state = force_torque_off(self.env, attempts)
        if _torque_is_off(state):
            self.engaged = False
        return state

    def engage(self, acceleration: int, speed: int, attempts: int = 3) -> None:
        """Enable torque holding the present pose. Nothing moves.

        Retried, because the servo bus fails transiently and this sequence is where it
        matters: a half-applied engage leaves torque indeterminate on an arm nobody is
        driving yet. Each failure forces torque off and verifies before trying again.
        """
        if self.no_motion:
            self.last_commanded = self.present()
            print("no-motion: torque NOT enabled, no action will be sent", flush=True)
            return

        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                self._engage_once(acceleration, speed)
                return
            except Abort:
                raise
            except Exception as exc:  # noqa: BLE001 - see below; the release is what matters
                # Every exception, not a list of the ones seen so far. A servo that answers with
                # an error byte raises RuntimeError, and a named-exception list let that escape
                # this loop — skipping the forced release between attempts, which is the reason
                # the loop exists at all.
                last_exc = exc
                print(
                    f"WARNING: engage attempt {attempt + 1}/{attempts} failed on the servo "
                    f"bus: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
                self.force_torque_off()
                time.sleep(0.5)
        raise Abort(f"could not enable torque safely after {attempts} attempts: {last_exc!r}")

    def _engage_once(self, acceleration: int, speed: int) -> None:
        # Torque off first, unless it is known to be off already: a previous run may have left
        # it holding a goal, in which case "present" would be that goal rather than a resting
        # pose. Skipping the provably-redundant case halves the writes in this sequence, and
        # this sequence is where the bus has been seen to drop one — `configure()` disables
        # again on its own way in, so nothing is lost. A read that came back empty is not
        # proof, so it writes.
        if not _torque_is_off(self.torque_state()):
            self.env.disable_torque()
        present = self.present()
        self.env.write_goal(present)

        # Re-enables torque on the way out, now onto a goal equal to the present pose.
        # Note it re-enables from a `finally`, so a failure inside it can leave torque on
        # — which is why the caller forces it off between attempts.
        self.env.configure()
        self.env.set_motion_limits(acceleration, speed)

        self.engaged = True
        self.last_commanded = present
        print(
            f"torque on, holding the present pose (acceleration={acceleration} speed={speed})",
            flush=True,
        )

    def send(self, pose: dict[str, float]) -> dict[str, float]:
        """Clamp to the envelope, suppress dither, and command. Returns what was sent.

        The deadband holds the previous command below a threshold, which reproduces the dwell
        the recordings have: this policy does not hold still, reversing direction on 24-68% of
        consecutive steps where the recordings reverse on under 1%.

        **It is off by default, and the reason is a retraction.** It was added to stop the
        gripper overheating, on a reading of 60 C after two minutes. Direct measurement later
        showed that is impossible: dithering the gripper harder than the policy ever did, for
        four minutes, moves it one degree. Real heating is about a quarter of a degree per
        minute, so 60 C would take hours. Every elevated reading was bus corruption.

        It also does not smooth the path — measured in end-effector space it changes nothing
        (49% of steps reverse before, 50% after), because each joint crosses its threshold at
        a different moment and the combination is a staircase.

        What it does do is reduce per-motor work, taking moving steps from 100% to 72%. That
        is real but was solving a non-problem, and it costs up to half the gripper's commanded
        travel, so it should be switched on for a measured reason rather than by default.
        """
        clamped = {joint: self.envelope.clamp(joint, pose[joint]) for joint in JOINTS}
        if self.last_commanded is not None:
            for joint in JOINTS:
                band = self.deadband_gripper if joint == GRIPPER else self.deadband_deg
                if abs(clamped[joint] - self.last_commanded[joint]) < band:
                    clamped[joint] = self.last_commanded[joint]
                    self.held[joint] += 1
        self.last_commanded = clamped
        if self.no_motion:
            return clamped
        sent = self.env.send_action({f"{j}.pos": v for j, v in clamped.items()})
        return {k.removesuffix(".pos"): v for k, v in sent.items()}

    def ramp_to(self, target: dict[str, float], seconds: float, hz: float = 30.0) -> None:
        """Interpolate to a pose over `seconds`, so a long move is not a lunge."""
        if self.no_motion or not self.engaged:
            return
        start = self.present()
        steps = max(1, int(seconds * hz))
        for i in range(1, steps + 1):
            frac = i / steps
            self.env.write_goal(
                {
                    joint: self.envelope.clamp(
                        joint, start[joint] + frac * (target[joint] - start[joint])
                    )
                    for joint in JOINTS
                },
            )
            time.sleep(1.0 / hz)
        self.last_commanded = target

    def settle(self, target: dict[str, float], timeout_s: float, tolerance: float = 2.0) -> float:
        """Wait for the arm to actually arrive. Returns the worst remaining error.

        Writing a goal is not reaching it. The ramp above finishes when it has *sent* the
        last goal, and with acceleration and speed capped for safety the arm is still
        travelling at that point — so dropping torque immediately leaves it wherever it
        got to, and makes "returns to the pose it started from" false.
        """
        deadline = time.perf_counter() + timeout_s
        worst = float("inf")
        while time.perf_counter() < deadline:
            now = self.present()
            worst = max(abs(now[j] - target[j]) for j in JOINTS)
            if worst <= tolerance:
                return worst
            time.sleep(0.05)
        return worst

    def release(self, home: dict[str, float] | None, seconds: float) -> None:
        """Return to `home` and drop torque. Must not raise, and must not be skipped.

        Built out of the environment's release-path members, each of which is specified to
        tolerate being called on a device that has already failed. That is the requirement:
        LeRobot's own ``disconnect()`` is gated on ``is_connected``, which includes the
        cameras, so routing a release through it would let a camera that dropped out raise —
        leaving the arm holding a pose, the failure this whole file is arranged around.

        **Torque-off is attempted whether or not this driver believes it engaged.** An
        engage that fails part-way leaves torque indeterminate, and that is exactly the
        moment a release must not be skipped — so only the *return* half is conditional on
        having engaged. In no-motion mode nothing is written at all, because the run
        already refused to start if torque was on.
        """
        if self.no_motion:
            return
        try:
            if self.engaged and home is not None:
                print("returning to the pose the interval started from...", flush=True)
                self.ramp_to(home, seconds)
                residual = self.settle(home, self.settle_timeout_s)
                self.return_residual = residual
                if residual <= 2.0:
                    print(f"back at the start pose (within {residual:.2f})", flush=True)
                else:
                    # Capped speed plus this arm's own shortfall, not a fault — but the
                    # operator should know the arm is not where it began.
                    print(
                        f"WARNING: stopped {residual:.1f} short of the start pose after "
                        f"{self.settle_timeout_s:g}s; releasing torque there",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001 - dropping torque matters more than this
            print(
                f"WARNING: return-to-start failed ({exc!r}); dropping torque now",
                file=sys.stderr,
                flush=True,
            )
        finally:
            state = self.force_torque_off()
            self.final_torque_state = state
            report_torque(state)


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
    only correct *briefly*, so when the retries are exhausted the run aborts and the
    arm is returned and released. The timeout behaviour is therefore: hold, retry, then
    release.
    """

    def __init__(self, host, port, *, camera_names, loop_timeout_ms, setup_timeout_s, retries):
        self._client_cls = _transport().PolicyClient
        self.host, self.port = host, port
        # Which frames go in an observation, and under what names. The server matches them
        # by name and never by position, so this is the whole of the camera contract here.
        self.camera_names = list(camera_names)
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

    def build_observation(self, obs: dict, instruction: str) -> dict:
        """Upstream's observation contract, over however many cameras are configured.

        ``float32`` is not incidental: the server validates dtypes and rejects anything
        else, which is the good outcome — the alternative is a silently rescaled state.

        The frame keys are the configured camera names, so a missing one is a `KeyError`
        naming the camera rather than a shape mismatch discovered on the server.
        """
        state = np.array([obs[key] for key in STATE_KEYS], dtype=np.float32)
        model_obs = {
            "video": {name: obs[name] for name in self.camera_names},
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
        MsgSerializer = _transport().MsgSerializer  # noqa: N806 - upstream's class name

        request = {
            "endpoint": "get_action",
            "data": {"observation": self.build_observation(obs, instruction), "options": None},
        }
        return len(MsgSerializer.to_bytes(request))

    def describe_observation(self, obs: dict, instruction: str) -> dict:
        """What the server will see, after the same msgpack round trip it performs.

        Encoded and decoded rather than introspected directly, because the interesting
        failures are ones where the wire format loses something — a key decoded as bytes
        instead of str, say, which reads on the server as a key that is simply absent.
        """
        MsgSerializer = _transport().MsgSerializer  # noqa: N806 - upstream's class name

        built = self.build_observation(obs, instruction)
        decoded = MsgSerializer.from_bytes(MsgSerializer.to_bytes(built))

        def describe(value):
            if isinstance(value, dict):
                return {f"{type(k).__name__}:{k!r}": describe(v) for k, v in value.items()}
            if isinstance(value, np.ndarray):
                return f"ndarray{value.shape} {value.dtype}"
            if isinstance(value, list):
                return [describe(v) for v in value[:1]] + (["..."] if len(value) > 1 else [])
            return f"{type(value).__name__}:{value!r}"[:60]

        return describe(decoded)

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
# Intervals
# ---------------------------------------------------------------------------


def run_interval(index, seconds, arm, policy, rate, args, trace, records) -> None:
    """Drive for ``seconds``, append the interval's record, and never leave the arm moving.

    An interval is not a trial and carries no verdict. It is a bounded stretch of
    torque-on driving that ends with the arm back at the pose it began from and limp,
    which is what makes it the unit worth having: the servos only cool while torque is
    off, and the policy expects to start near the pose it was trained from.

    The record is appended in a ``finally`` rather than returned, so an abort halfway
    through still publishes what happened up to that point — which is the interesting
    part when something went wrong.
    """
    start_pose = arm.present()
    state = {
        "ended": "completed", "note": "",
        # None, not 0: an environment that does not report heat has no peak, and a zero here
        # would read as a measurement of a cool arm.
        "steps": 0, "replans": 0, "max_temp_c": None, "hottest_temps": {}, "ramp_s": 0.0,
    }
    # Commanded-versus-achieved, sampled once per chunk. This arm does not reach what it
    # is told — unevenly, per joint — so the gap is a first-class result, not a detail.
    achieved_gap: dict[str, list[float]] = {j: [] for j in JOINTS}
    # The interval's own clock starts here, before the ramp, so `seconds` bounds the whole
    # evaluation rather than only the driven part of it.
    interval_start = time.perf_counter()

    try:
        if args.home_pose:
            print(f"interval {index}: ramping to the home pose over {args.home_ramp_s}s", flush=True)
            arm.ramp_to(args.home_pose, args.home_ramp_s)
        # Timed separately so the control rate can be reported over the driven part alone.
        # Dividing by the whole interval understates it by however long the ramp took.
        state["ramp_s"] = time.perf_counter() - interval_start

        rate.reset()
        while True:
            if _stop.is_set():
                raise Abort("stop requested")
            if (time.perf_counter() - interval_start) >= seconds:
                state["ended"] = "elapsed"
                break

            # Once per re-plan, which is cheap next to a round trip. A servo that reaches
            # its own Max_Temperature_Limit cuts its torque and drops the arm wherever it
            # is; aborting here trades that for a controlled return while the arm is still
            # holding. Joints carrying gravity load heat fastest, so this is what bounds a
            # long session rather than the clock.
            temps = arm.temperatures()
            if temps is None:
                # No heat to manage in this environment. Distinct from the empty case below, and
                # `max_temp_c` stays None rather than reporting a cool arm it never measured.
                pass
            elif not temps:
                # Every joint's reading was rejected: the bus is unhealthy, not the arm hot.
                arm.hot_streak = 0
            else:
                hottest = max(temps.values())
                if state["max_temp_c"] is None or hottest > state["max_temp_c"]:
                    state["max_temp_c"] = hottest
                    # Which joint, not just how hot. The maximum alone does not say where
                    # the heat is, and on this arm it has been one joint at a time.
                    state["hottest_temps"] = dict(temps)
                # Confirmed over consecutive reads, because one bad packet is not a hot
                # servo. Real heating persists across reads; corruption does not, and
                # believing a single read cost three rests in one run for nothing.
                if hottest >= args.max_servo_temp_c:
                    arm.hot_streak += 1
                    if arm.hot_streak >= args.temp_confirm_reads:
                        raise Cooldown(
                            f"servo temperature {hottest}C reached the "
                            f"{args.max_servo_temp_c}C ceiling on "
                            f"{arm.hot_streak} consecutive reads ({temps}); resting before "
                            f"the servos cut their own torque"
                        )
                    print(
                        f"NOTE: {hottest}C seen at or above the {args.max_servo_temp_c}C "
                        f"ceiling ({arm.hot_streak}/{args.temp_confirm_reads} reads); "
                        f"confirming before resting",
                        flush=True,
                    )
                else:
                    arm.hot_streak = 0

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
                                "interval": index,
                                "step": state["steps"],
                                "t": round(time.perf_counter() - interval_start, 4),
                                "requested": {j: round(v, 3) for j, v in target.items()},
                                "sent": {j: round(v, 3) for j, v in last_sent.items()},
                            }
                        )
                        + "\n"
                    )
                rate.sleep()
                if _stop.is_set():
                    raise Abort("stop requested")
                if (time.perf_counter() - interval_start) >= seconds:
                    state["ended"] = "elapsed"
                    break

            if last_sent is not None and not arm.no_motion:
                reached = arm.present()
                for joint in JOINTS:
                    achieved_gap[joint].append(reached[joint] - last_sent[joint])
            if state["ended"] == "elapsed":
                break
    except Cooldown as exc:
        state["ended"] = "cooldown"
        state["note"] = str(exc)
        raise
    except Abort as exc:
        state["ended"] = "aborted"
        state["note"] = str(exc)
        raise
    finally:
        duration = time.perf_counter() - interval_start
        driven = duration - state["ramp_s"]
        records.append(
            {
                "interval": index,
                "ended": state["ended"],
                "note": state["note"],
                "duration_s": round(duration, 2),
                "ramp_s": round(state["ramp_s"], 2),
                "driven_s": round(driven, 2),
                "steps": state["steps"],
                "replans": state["replans"],
                # Over the driven part only. Including the ramp would report a control rate
                # the loop never ran at.
                "achieved_hz": round(state["steps"] / driven, 2) if driven > 0 else None,
                "max_temp_c": state["max_temp_c"],
                "hottest_temps": state["hottest_temps"],
                "start_pose": {j: round(v, 3) for j, v in start_pose.items()},
                "mean_shortfall": {
                    j: round(statistics.fmean(v), 3) for j, v in achieved_gap.items() if v
                },
            }
        )
        # Return to where this interval began, then release, whatever happened.
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
    g.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        metavar="NAME=SPEC",
        help="Repeatable, once per camera. NAME is the dataset's image-feature suffix and "
        "is what the policy server matches on. SPEC is either a selector resolved against "
        "sysfs on this machine — 'sysfs:<glob of the device path>' or 'name:<glob of the "
        "v4l2 card name>', each of which must match exactly one capture device — or any "
        "value OpenCV accepts, passed through untouched: a device node, an index, a video "
        "file, a URL. Prefer a selector: /dev/videoN is assigned in enumeration order, so "
        "it changes on replug and fails silently by opening the wrong camera.",
    )
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
    g.add_argument(
        "--command-deadband-deg",
        type=float,
        default=0.0,
        help="Hold the previous command when an arm joint's target moves less than this. "
        "OFF by default: it was added to cut servo heating, and direct measurement showed "
        "the heating was never real. It also does not smooth the path. Available because it "
        "genuinely reduces per-motor work, but it discards commanded travel, so enable it "
        "only for a reason you have measured.",
    )
    g.add_argument("--command-deadband-gripper", type=float, default=0.0)
    g.add_argument(
        "--max-servo-temp-c",
        type=int,
        default=55,
        help="Pause and rest if any servo reaches this. Their Max_Temperature_Limit reads "
        "70 C with overheat protection enabled, and the vendor table says protection derates "
        "output to Protective_Torque -- 20%% of maximum here, not zero. This stops short of "
        "that, while the arm is still fully driven.",
    )
    g.add_argument(
        "--temp-confirm-reads",
        type=int,
        default=3,
        help="Consecutive over-ceiling readings needed before resting. One corrupted read "
        "is not a hot servo; real heating persists across reads.",
    )
    g.add_argument(
        "--resume-servo-temp-c",
        type=int,
        default=45,
        help="Resume once every servo is back under this. Deliberately below the pause "
        "threshold: without the gap the run would stop and start every few seconds.",
    )
    g.add_argument(
        "--cooldown-timeout-s",
        type=float,
        default=600.0,
        help="Give up if the servos will not cool within this. Failing to cool with torque "
        "off means something is still drawing current.",
    )
    g.add_argument(
        "--max-cooldowns",
        type=int,
        default=20,
        help="Stop after this many rests rather than cycling indefinitely on an arm that "
        "cannot sustain the duty cycle.",
    )
    g.add_argument("--release-ramp-s", type=float, default=2.0)
    g.add_argument(
        "--release-settle-s",
        type=float,
        default=6.0,
        help="How long to wait for the arm to actually arrive at the start pose before "
        "dropping torque. Sending the last goal is not reaching it, and --speed caps how "
        "fast it can get there.",
    )

    g = p.add_argument_group("the control loop")
    g.add_argument("--lang-instruction", required=True)
    g.add_argument("--action-horizon", type=int, default=8)
    g.add_argument("--control-hz", type=float, default=30.0)
    g.add_argument(
        "--run-duration-s",
        type=float,
        default=20.0,
        help="How long the evaluation runs in total, counting the home ramp at the start "
        "of every interval. The only knob for run length.",
    )
    g.add_argument(
        "--interval-s",
        type=float,
        default=60.0,
        help="How long the arm drives before returning to its starting pose and "
        "releasing torque, then starting again. Sets how often the servos get to cool "
        "and how often you get a limp arm to re-stage the scene with; it does not "
        "change how long the run lasts.",
    )
    g.add_argument(
        "--home-pose",
        default=None,
        help="Six values in JOINTS order. When given, every interval ramps here first, so "
        "the policy always starts from the pose it was trained from rather than from "
        "wherever the last one left off.",
    )
    g.add_argument("--home-ramp-s", type=float, default=4.0)

    g.add_argument(
        "--transport-module",
        default=TRANSPORT_MODULE,
        help="Importable module providing PolicyClient and MsgSerializer. Defaults to the "
        "reference implementation; a different policy server can supply its own so this "
        "client need not be duplicated to serve it.",
    )

    g = p.add_argument_group("output")
    g.add_argument("--output-dir", required=True)
    return p


def main() -> int:  # noqa: PLR0915 - a linear bring-up sequence, ordered by risk
    args = build_parser().parse_args()
    global TRANSPORT_MODULE  # noqa: PLW0603 - set once, before anything imports it
    TRANSPORT_MODULE = args.transport_module
    args.home_pose = _parse_pose(args.home_pose)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # The one line that names an implementation. Everything after it is the interface, so running
    # against a simulator means a different module injected beside this file.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from so101_environment import SO101Environment  # noqa: PLC0415 - injected beside this file

    # A cancelled workflow arrives as SIGTERM, which is the operator's way to stop a run
    # early: it unwinds through the same return-and-release path a timeout does.
    _install_signal_handlers()

    # Before anything else, because a malformed --camera is a usage error: there is no arm
    # to release yet, and failing here costs nothing.
    try:
        camera_specs = _parse_cameras(args.cameras)
    except Abort as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2

    records: list[dict] = []
    results = {
        "config": {k: v for k, v in vars(args).items()},
        "intervals": records,
    }
    arm = None
    env = None
    envelope = None
    rate = RateLimiter(args.control_hz)

    policy = PolicyLink(
        args.policy_host,
        args.policy_port,
        camera_names=list(camera_specs),
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

        # ---- 2. the environment, selected but not opened ---------------------
        # Constructing it picks the hardware — the port, the calibration file, and which capture
        # device each camera name means — and opens nothing, so a stale camera selector or a
        # missing calibration file fails here with the arm untouched.
        try:
            env = SO101Environment(
                port=args.robot_port,
                robot_id=args.robot_id,
                camera_specs=camera_specs,
                camera_width=args.camera_width,
                camera_height=args.camera_height,
                camera_fps=args.camera_fps,
                calibration_dir=args.calibration_dir or None,
            )
        except ValueError as exc:
            raise Abort(str(exc)) from exc

        # What each camera name actually reads, not what it was asked to read. Published, because
        # a name alone does not say which device produced the frames a result came from.
        results["cameras"] = env.camera_sources()
        for name, source in results["cameras"].items():
            print(f"camera {name}: {source}", flush=True)

        # ---- 3. open it ------------------------------------------------------
        # connect() makes every member usable and does no more: LeRobot's own connect() would run
        # configure(), which enables torque, before any of the checks below can refuse.
        env.connect()
        print(
            f"environment connected, {len(results['cameras'])} camera(s) open",
            flush=True,
        )

        # Asked unconditionally: whether the question means anything is the environment's answer
        # to give, not something to infer from the flags this run was started with.
        matches = env.is_calibrated()
        if matches is None:
            print("no calibration was claimed; the device's own frame is in use", flush=True)
        elif not matches:
            raise Abort(
                "the device's stored calibration does not match the calibration file, so "
                "the arm is not in the joint frame the dataset was recorded in — refusing "
                "to drive it"
            )
        else:
            print("stored calibration matches the calibration file", flush=True)

        # In the units actions arrive in. Whatever the device stores internally is its own
        # business and never reaches here.
        try:
            limits = env.joint_limits()
        except ValueError as exc:
            # The environment refusing its own limits is an operator-facing abort, not a crash:
            # a traceback here would bury the one sentence that says what to re-calibrate.
            raise Abort(f"the environment reports unusable joint limits: {exc}") from exc
        problems = sanity_check_limits(limits)
        if problems:
            raise Abort("reported joint limits are not usable: " + "; ".join(problems))
        results["joint_limits"] = {j: [round(v, 2) for v in lim] for j, lim in limits.items()}
        print("joint limits reported by the environment:", flush=True)
        for joint, (lo, hi) in limits.items():
            print(f"  {joint:<14} {lo:>8.2f} .. {hi:>8.2f}", flush=True)

        envelope = SafetyEnvelope.from_limits(
            limits, args.joint_margin_deg, args.gripper_margin
        )
        results["envelope"] = {j: [round(v, 2) for v in lim] for j, lim in envelope.limits.items()}
        print("safety envelope, those limits less the margin:", flush=True)
        for joint, (lo, hi) in envelope.limits.items():
            print(f"  {joint:<14} {lo:>8.2f} .. {hi:>8.2f}", flush=True)

        # The other half of the clamp: the envelope bounds where a command may be, this bounds
        # how far one may move from where the arm is. Enforced inside the environment, against a
        # position it reads itself, so it holds for whatever the policy emits.
        step_limit = {**{j: args.max_step_deg for j in ARM_JOINTS}, GRIPPER: args.max_step_gripper}
        env.set_step_limit(step_limit)
        results["step_limit"] = step_limit
        print(
            f"per-step limit: {args.max_step_deg} deg per arm joint, "
            f"{args.max_step_gripper} for the gripper",
            flush=True,
        )

        arm = ArmDriver(
            env,
            envelope,
            no_motion=args.no_motion,
            settle_timeout_s=args.release_settle_s,
            deadband_deg=args.command_deadband_deg,
            deadband_gripper=args.command_deadband_gripper,
        )
        results["is_physical"] = env.is_physical()

        entry_torque = arm.torque_state()
        results["entry_torque"] = entry_torque
        if entry_torque == {}:
            print(
                "WARNING: could not read whether torque is already enabled; proceeding as if "
                "it is not, so engage will drop it before re-enabling onto the present pose",
                file=sys.stderr,
                flush=True,
            )
        energised = [j for j, v in (entry_torque or {}).items() if v]
        if energised and args.no_motion:
            # no-motion promises not to write to the bus at all, so it cannot fix this —
            # and it must not proceed while the arm is holding a pose it did not command.
            raise Abort(
                f"torque is already enabled on {energised}; the arm is energised from an "
                f"earlier run. Release it before a no-motion rehearsal, which must not "
                f"write to the servo bus"
            )
        if energised:
            print(
                f"NOTE: torque was already enabled on {energised} at start; engage will "
                f"drop it before re-enabling onto the present pose",
                flush=True,
            )

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

        # ---- 4. warm-up, with the arm still limp ------------------------------
        # The whole policy path — observation shape, dtypes, server-side validation, chunk
        # length, and whether the returned actions are even inside the arm's reachable
        # range — is exercised here, before torque. If any of it is wrong, the run ends
        # with the arm untouched.
        #
        # Through the environment, like every other observation. Reading the same data by a
        # second route is how the warm-up comes to validate a path the loop does not take.
        obs = env.get_observation()
        payload = policy.payload_bytes(obs, args.lang_instruction)
        results["observation_bytes"] = payload
        print(
            f"observation payload: {payload / 1e6:.2f} MB per request "
            f"({args.camera_width}x{args.camera_height} raw uint8 x{len(camera_specs)})",
            flush=True,
        )
        # The shape actually going on the wire, as the server will see it after a msgpack
        # round trip. A server-side "must be in observation" rejection names a key without
        # saying what it did receive, so record that here instead of inferring it.
        results["observation_shape"] = policy.describe_observation(obs, args.lang_instruction)
        print(f"observation as the server receives it: "
              f"{json.dumps(results['observation_shape'])}", flush=True)
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

        # ---- 5. the control loop ------------------------------------------------
        policy.close()
        policy.open_for_loop()
        print(f"switching to the control-loop timeout of {args.policy_timeout_ms} ms", flush=True)

        # The run is bounded by one number: how long the evaluation should last. Intervals
        # repeat inside that budget and are not counted, because their number is not a
        # result — they exist so the arm periodically returns home and goes limp, which is
        # the only time the servos cool and the only chance to re-stage the scene by hand.
        #
        # The budget counts each interval whole, including its home ramp, because it bounds
        # how long the evaluation runs rather than how much of that was driving. Resting
        # does not count, so a hot arm makes the run last longer in wall clock instead.
        rests: list[dict] = []
        results["rests"] = rests
        print(
            f"evaluating for {args.run_duration_s:g}s in intervals of at most "
            f"{args.interval_s:g}s",
            flush=True,
        )
        with (output / "trace.jsonl").open("w") as trace:
            index, spent = 0, 0.0
            while spent < args.run_duration_s:
                index += 1
                # Never overshoot the budget: the last interval is whatever is left.
                seconds = min(args.interval_s, args.run_duration_s - spent)
                arm.engage(args.acceleration, args.speed)
                try:
                    run_interval(index, seconds, arm, policy, rate, args, trace, records)
                except Cooldown as exc:
                    # run_interval's own teardown has already returned the arm and dropped
                    # torque, which is the precondition for cooling at all. The time it used
                    # before getting hot still counts against the budget.
                    spent += records[-1]["duration_s"]
                    print(f"interval {index}: {exc}", flush=True)
                    if len(rests) >= args.max_cooldowns:
                        raise Abort(
                            f"reached the {args.max_cooldowns}-rest limit; the arm cannot "
                            f"sustain this duty cycle, so stopping rather than cycling"
                        ) from exc
                    rests.append(
                        arm.wait_to_cool(args.resume_servo_temp_c, args.cooldown_timeout_s)
                        | {"interval": index}
                    )
                    continue
                spent += records[-1]["duration_s"]
                peak = records[-1]["max_temp_c"]
                heat = f"hottest {peak}C, " if peak is not None else ""
                print(
                    f"interval {index}: {records[-1]['ended']} "
                    f"({records[-1]['steps']} steps at {records[-1]['achieved_hz']} Hz over "
                    f"{records[-1]['driven_s']:.0f}s driving, "
                    f"{heat}"
                    f"{spent:.0f}/{args.run_duration_s:g}s used)",
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
        final_state: dict[str, int] | None = {}
        if arm is not None:
            arm.release(arm.last_commanded, args.release_ramp_s)
            final_state = arm.final_torque_state
        elif env is not None and not args.no_motion:
            # Connected, but bring-up failed before there was a driver — a camera that would not
            # open, unusable limits, a refused calibration. Nothing here drove the arm, but it may
            # have arrived energised from an earlier run, and that is exactly when a release must
            # not be skipped. No return ramp: nothing knows where this run would have started.
            final_state = force_torque_off(env)
            report_torque(final_state)
        # After the release, never before it: closing the environment is housekeeping and
        # dropping torque is not. disconnect() is specified to tolerate a partly-open
        # environment, which is what this path always has.
        if env is not None:
            try:
                env.disconnect()
            except Exception as exc:  # noqa: BLE001 - teardown, and torque is already off
                print(f"WARNING: disconnect failed: {exc!r}", file=sys.stderr, flush=True)
        policy.close()

        # No success field, and no count of anything, by design. Whether the policy did
        # something useful is the operator's call, made at the bench while it runs; there is
        # no sensor for it and a number here would only look like one. What belongs here is
        # what the eye cannot measure: timing, reach, and heat.
        results["summary"] = {
            "chunk_latency": _percentiles(policy.latencies_ms),
            "policy_timeouts": policy.timeouts,
            "loop_overruns": rate.overruns,
            "worst_loop_overrun_ms": round(rate.worst_overrun_ms, 1),
            "clipped_by_joint": dict(envelope.clipped) if envelope else {},
            "held_by_deadband": dict(arm.held) if arm else {},
            "no_motion": args.no_motion,
            # Read back off the servos, not inferred from having asked. This is the field
            # to check first after any run that ended badly.
            "final_torque_state": final_state,
            "return_residual": arm.return_residual if arm else None,
            # None when nothing was ever measured, rather than a zero standing in for one.
            "max_temp_c": max(
                (e["max_temp_c"] for e in records if e.get("max_temp_c") is not None),
                default=None,
            ),
            "rests": len(results.get("rests") or []),
            "bad_temp_reads": arm.bad_temp_reads if arm else 0,
            "temp_peak_accepted": (
                max((max(v for k, v in r.items() if k != "t") for r in arm.temp_log), default=None)
                if arm and arm.temp_log else None
            ),
            "rested_s": round(sum(r["rested_s"] for r in (results.get("rests") or [])), 1),
            # env is None means nothing was ever opened, so there was nothing to release. An arm
            # that WAS opened is judged on the read-back, whether or not it got as far as driving.
            "torque_released": bool(args.no_motion or env is None) or _torque_is_off(final_state),
        }
        # The whole temperature series, not just a maximum. A maximum over hundreds of
        # reads is inflated by any single bad one, which is precisely how an unconfirmed
        # thermal problem came to be treated as established.
        if arm is not None and arm.temp_log:
            with (output / "temperatures.jsonl").open("w") as handle:
                for row in arm.temp_log:
                    handle.write(json.dumps(row) + "\n")
        (output / "results.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
        print("\n" + json.dumps(results["summary"], indent=2), flush=True)

    return 1 if results.get("aborted") else 0


if __name__ == "__main__":
    sys.exit(main())
