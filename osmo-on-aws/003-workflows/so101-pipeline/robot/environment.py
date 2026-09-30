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
"""What an SO-101 is driven against: a real arm, or a simulator.

The robot's contract, not the workflow's. The shape is LeRobot's `Robot` — connect,
get_observation, send_action — plus the operations an arm's safety behaviour needs and `Robot`
does not expose.

`connect` and `disconnect` bracket everything else, and that is the ONLY ordering this interface
imposes. Between them EVERY OTHER MEMBER IS TOTAL: each may be invoked at any time, in any order,
any number of times, and no member is conditional on another having been called. `connect`
therefore has to establish whatever the rest needs — a calibration to convert units with, an open
sensor — because after it returns there is no second setup step to depend on.

An environment for which a member has no meaning says so in its own return value:

  * An OPERATION for which there is nothing to do does nothing. It is the environment's judgement
    whether there is something to do — `configure` on a simulator may reset an articulation where
    on real servos it enables torque, and neither is more correct.
  * A QUERY with no answer returns None, which is distinct from a failed reading. None means "this
    environment has no such notion"; an empty result means "I should be able to tell you and
    could not". Those demand opposite responses, so they must not share a representation.

A query MUST NOT answer None's opposite: a plausible constant. Reporting 25 C forever is worse
than reporting nothing, because nothing is checkable and 25 C is not.

An operation MAY raise — the device it speaks to can fail. It MUST be idempotent, and a failure
MUST leave the device no more energised than before the call. No operation is confirmed by having
returned: torque is proved off by reading it back, which is why `disable_torque` and `torque_state`
are separate members.

`is_physical` is provenance. Nothing branches on it.

Design rationale, alternatives weighed, and the measured limit differences between this arm and a
simulated one are in the ticket, not here.
"""

from __future__ import annotations

import abc


class Environment(abc.ABC):
    """An SO-101, real or simulated. See the module docstring for the totality rule.

    Every member is a method, including the ones that read like attributes. Do not add a property:
    `if env.is_calibrated:` is truthy forever.
    """

    @abc.abstractmethod
    def is_physical(self) -> bool:
        """True for real hardware. PROVENANCE ONLY — behaviour must not depend on it.

        What an environment cannot do is expressed by the member concerned, never inferred here.
        """

    # --- the bracket ----------------------------------------------------------------------
    @abc.abstractmethod
    def connect(self) -> None:
        """Make every other member usable, and do no more than that.

        MUST NOT begin holding position — that is `configure`.

        MUST leave nothing for a caller to finish: open the sensors `get_observation` reads,
        establish the unit mapping `send_action` and `joint_limits` need. After this returns, no
        member may still be waiting on some other member to have run.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close what `connect` opened. Called while unwinding, possibly after a failure.

        MUST tolerate being called on a partly-open environment, and MUST NOT be gated on
        everything having stayed healthy: a dropped camera must not stop the arm's bus being
        closed.
        """

    # --- the control loop -----------------------------------------------------------------
    @abc.abstractmethod
    def get_observation(self) -> dict:
        """Joint positions as `<joint>.pos` plus one frame per camera, keyed by camera name."""

    @abc.abstractmethod
    def camera_sources(self) -> dict[str, str]:
        """Camera name -> what is ACTUALLY being read under that name, for the record.

        The names are fixed by the dataset and go on the wire; what sits behind each one is the
        environment's business — a capture device it selected, a render target, a video file. Say
        which, in whatever terms are true here, because "the run used the camera called front" is
        not a checkable statement, and naming the device and the port it was found on is.

        Answerable before `connect`, so a mis-aimed camera is a refusal rather than a bad frame.
        `{}` if this environment has no cameras.
        """

    @abc.abstractmethod
    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Apply `<joint>.pos` targets and return what was ACTUALLY applied.

        The returned value MUST be what was applied, not what was requested. Those differ
        whenever the environment limits a command, and the difference is the only record that the
        limiting happened. Returning the input unchanged asserts that nothing was limited.
        """

    @abc.abstractmethod
    def present_position(self) -> dict[str, float]:
        """Where the joints ACTUALLY are, in the units `send_action` takes.

        MUST be the achieved state, never an echo of the last command.
        """

    @abc.abstractmethod
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Per-joint travel limits, IN THE UNITS `send_action` AND `get_observation` USE.

        Not device-native units: any conversion happens behind this interface. For a real arm
        these come from its calibration sweep, for a simulator from its model, and the two differ —
        report your own, never a constant copied from another environment.

        Raises ValueError, naming the joints, if the environment can tell its own limits are
        unusable. On real servos a range spanning the whole encoder means the sweep never captured
        that joint; only the device recognises its own version of that. This member refuses instead
        of returning None because there is no driving the arm without limits, whereas a None
        elsewhere means "carry on without this one thing".
        """

    @abc.abstractmethod
    def write_goal(self, pose: dict[str, float]) -> None:
        """Target `pose` exactly, with NO per-step limit applied.

        The one difference from `send_action`: that applies `set_step_limit`, this does not. The
        environment's own travel limits still apply, so this is not a way past `joint_limits`.
        """

    # --- limits the environment enforces on the caller's behalf ---------------------------
    # Set once, then applied by the environment on every command until changed.

    @abc.abstractmethod
    def set_step_limit(self, per_joint: dict[str, float]) -> None:
        """Cap, per joint, how far one `send_action` may move from the PRESENT position.

        Against the present position, not the last command: a joint that fell short of its last
        target must not accumulate permission to jump.

        MUST be idempotent. MUST apply to `send_action`, MUST NOT apply to `write_goal`.
        """

    @abc.abstractmethod
    def set_motion_limits(self, acceleration: int, speed: int) -> None:
        """Cap how fast the device may travel toward a target, so a wrong one is approached
        slowly rather than snapped to. Servo registers on real hardware; a velocity and effort
        limit on a simulated articulation.

        MUST be idempotent. An environment with no adjustable approach speed does nothing.

        Some devices reset these when made ready, so `configure` may clear them.
        """

    # --- queries that an environment may have no notion of --------------------------------
    # Each returns None for "no such notion here". None is NOT the same as an empty or failed
    # reading, and the two demand opposite responses, so keep them distinct.

    @abc.abstractmethod
    def temperatures(self) -> dict[str, int] | None:
        """Per-joint temperature, or None if this environment does not model heat.

        Unfiltered — report what was read, including values you suspect. Rejecting implausible
        ones happens above, where the transport's corruption pattern is known.

        `{}` means the read was attempted and yielded nothing, which is a fault. None means there
        is nothing to read, which is not.
        """

    @abc.abstractmethod
    def torque_state(self) -> dict[str, int] | None:
        """Per-joint torque-enable, READ BACK from the device, never remembered from a write.

        None if this environment has no notion of torque being on or off. A device that resists
        being moved has that notion, however it is implemented.

        `{}` means the state could not be read, and MUST NOT be returned in place of None.
        "Nothing is holding" and "I cannot tell whether anything is holding" are opposite answers,
        and this is the question the release path turns on.
        """

    @abc.abstractmethod
    def is_calibrated(self) -> bool | None:
        """Is the device in the joint frame that was claimed for it?

        A calibration supplied when the environment was constructed is a claim about which frame
        the recordings were made in. True means the device agrees with that claim.

        None means no claim was made, so the frame is whatever the device itself reports and there
        is nothing to disagree with. False MUST mean "they differ" and never "I could not tell" —
        a doubtful False is indistinguishable from a real mismatch, and False refuses to drive.
        """

    # --- unwinding ------------------------------------------------------------------------
    @abc.abstractmethod
    def disable_torque(self) -> None:
        """Stop holding position — let the joints go slack.

        On real servos, torque off. Elsewhere, whatever makes the device stop resisting: dropping
        actuator stiffness, for instance. A device that never resists does nothing.

        MUST be idempotent, and MUST NOT be conditional on torque currently being on, on the
        device being healthy, or on an earlier call having succeeded — this runs repeatedly while
        unwinding, after something has already gone wrong. It MAY raise, and MUST NOT leave the
        device holding more firmly than it was. Returning is not proof; `torque_state` is.
        """

    @abc.abstractmethod
    def configure(self) -> None:
        """Make the device ready to be commanded.

        On real servos this is where torque comes on; a simulator may reset its articulation or
        set gains. An environment that needs no preparation does nothing.

        MUST be idempotent, and MUST NOT choose a goal of its own: if this establishes a hold, the
        hold is against whatever goal `write_goal` last set. A stale goal here is a lunge.

        MAY clear what `set_motion_limits` set, on a device where becoming ready rewrites those
        registers.
        """
