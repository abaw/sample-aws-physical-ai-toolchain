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
"""The real SO-101, behind the environment interface. Mostly forwarding to LeRobot's
`SO101Follower`; the safety behaviour lives above, in the client's driver.

Everything Linux- and LeRobot-specific about reaching this arm lives here: the serial port, the
calibration file, and turning a camera spec into a capture device. None of it is meaningful to a
simulator, so none of it belongs in the client.

Do not call `SO101Follower.connect()` from here. It runs `configure()` on its way out, which
re-enables torque onto whatever `Goal_Position` was last written — a previous run's, or one from
before the arm was moved by hand. `connect()` below opens the bus and the cameras and stops there,
leaving `configure()` a separate call.

Failures raise ValueError naming the spec or path at fault. This module does not import the
client, so it cannot raise the client's `Abort`; the client converts.
"""

from __future__ import annotations

import fnmatch
import os
import pathlib

from environment import Environment

# A camera is named by the dataset's image-feature suffix and pointed at by a spec. /dev/videoN is
# assigned in enumeration order, so a replug or a reboot renumbers it: a spec naming one directly
# goes stale without saying so, and the arm then faithfully drives on the wrong camera's frames.
# The selector forms name something durable instead. sysfs is host-wide and mounted into every
# container by the runtime, so resolving needs no volumeMount and no privileged.
V4L2_SYSFS = pathlib.Path("/sys/class/video4linux")


def _v4l2_nodes() -> list[dict]:
    """Every v4l2 node the kernel knows about, whatever bus it sits on.

    A node is identified by its **major:minor**, not by its name: the number in ``videoN`` is the
    thing being distrusted, so matching the container's ``/dev`` entry by name would reintroduce
    the assumption the selector exists to remove.
    """
    nodes = []
    for entry in sorted(V4L2_SYSFS.glob("video*")):
        try:
            major, minor = (int(x) for x in (entry / "dev").read_text().strip().split(":"))
            nodes.append(
                {
                    "sysfs": str(entry.resolve()),
                    "name": (entry / "name").read_text().strip(),
                    "index": int((entry / "index").read_text().strip()),
                    "rdev": os.makedev(major, minor),
                }
            )
        except (OSError, ValueError):
            # A node that disappeared mid-scan, or one whose attributes a driver does not publish.
            # Skipping is right: it cannot be selected either way.
            continue
    return nodes


def _device_for_rdev(rdev: int) -> pathlib.Path | None:
    """The ``/dev`` entry with this major:minor, or None if it was not granted to us."""
    for candidate in sorted(pathlib.Path("/dev").glob("video*")):
        try:
            if os.stat(candidate).st_rdev == rdev:
                return candidate
        except OSError:
            continue
    return None


def _resolve_camera_spec(spec: str) -> tuple[object, str]:
    """One camera spec into something ``cv2.VideoCapture`` accepts.

    Returns ``(index_or_path, account)``, where the account says which physical device was
    actually selected rather than which one was asked for.
    """
    for prefix, field in (("sysfs:", "sysfs"), ("name:", "name")):
        if not spec.startswith(prefix):
            continue
        pattern = spec[len(prefix) :]
        if not pattern:
            raise ValueError(f"camera spec {spec!r} has an empty pattern")

        # index 0 is the capture node on a UVC camera and the others carry metadata, so this
        # filter is what stops a two-node camera matching itself twice. It is a convenience, not a
        # guarantee: a wrong pick would fail in cv2 at connect() anyway.
        capture = [n for n in _v4l2_nodes() if n["index"] == 0]
        matches = [n for n in capture if fnmatch.fnmatch(n[field], pattern)]
        if not matches:
            seen = ", ".join(repr(n[field]) for n in capture) or "none"
            raise ValueError(f"camera spec {spec!r} matched no capture device. Present: {seen}")
        if len(matches) > 1:
            raise ValueError(
                f"camera spec {spec!r} is ambiguous, matching "
                + ", ".join(m["sysfs"] for m in matches)
                + " — refusing to guess which one the dataset means"
            )

        node = matches[0]
        device = _device_for_rdev(node["rdev"])
        if device is None:
            raise ValueError(
                f"camera spec {spec!r} resolved to {node['sysfs']}, but no /dev entry with that "
                f"major:minor is visible here"
            )
        return device, f"{spec} -> {device} ({node['name']}, {node['sysfs']})"

    # Not a selector, so it is OpenCV's own vocabulary: an index, a device node, a video file, or a
    # URL. Deliberately not re-validated — cv2.VideoCapture accepts more forms than a check here
    # would, and narrowing it is how the /dev/video2 assumption got in.
    if spec.isdigit():
        return int(spec), f"{spec} -> OpenCV index {spec} (verbatim)"
    if spec.startswith("/") or pathlib.Path(spec).exists():
        return pathlib.Path(spec), f"{spec} -> {spec} (verbatim, not resolved)"
    return spec, f"{spec} -> {spec} (verbatim, handed to OpenCV as-is)"


class SO101Environment(Environment):
    """A physical SO-101 follower, driven through LeRobot's `SO101Follower`.

    A `calibration_dir` is the claim `is_calibrated()` checks the servos against, and is left
    exactly as supplied. With none, the servos' EEPROM is read at `connect()` and becomes the
    frame, and `is_calibrated()` returns None.

    `SO101Follower` is an alias of LeRobot's `SOFollower`, as is `SO100Follower`, so nothing here
    distinguishes the two arms by type. What makes this an SO-101 is the calibration on the bus and
    the six joint names.
    """

    # STS3215 has 4096 encoder positions, so the largest index is 4095.
    _full_span = 4095

    def __init__(
        self,
        *,
        port: str,
        robot_id: str,
        camera_specs: dict[str, str],
        camera_width: int,
        camera_height: int,
        camera_fps: int,
        calibration_dir: str | None = None,
    ):
        """Select the hardware and build the LeRobot objects. Opens nothing.

        Camera specs are resolved here rather than in `connect()`, so a stale selector fails while
        the arm is still untouched. `camera_specs` maps the dataset's image-feature name to a
        selector or an OpenCV-accepted value; see `_resolve_camera_spec`.

        `calibration_dir` must contain `<robot_id>.json` if given. Checked here because LeRobot
        silently falls back to the servos' own EEPROM for a missing file, and the run would then
        proceed in whatever frame the arm happens to hold.
        """
        # Imported here, not at module scope: this module is importable without LeRobot so the
        # interface can be exercised off the bench, and only this method needs the real package.
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        if calibration_dir is not None:
            path = pathlib.Path(calibration_dir) / f"{robot_id}.json"
            if not path.is_file():
                raise ValueError(
                    f"calibration directory {calibration_dir} contains no {robot_id}.json; "
                    f"LeRobot would fall back to the servos' EEPROM without saying so"
                )

        self._sources: dict[str, str] = {}
        devices: dict[str, object] = {}
        for name, spec in camera_specs.items():
            devices[name], self._sources[name] = _resolve_camera_spec(spec)

        robot = SO101Follower(
            SO101FollowerConfig(
                port=port,
                id=robot_id,
                calibration_dir=pathlib.Path(calibration_dir) if calibration_dir else None,
                cameras={
                    name: OpenCVCameraConfig(
                        index_or_path=device,
                        fps=camera_fps,
                        width=camera_width,
                        height=camera_height,
                    )
                    for name, device in devices.items()
                },
            )
        )
        self._robot = robot
        self._bus = robot.bus
        # Whether the frame came from outside or off the servos. Decided in connect(), where the
        # servos can first be read.
        self._calibration_claimed: bool | None = None

    def is_physical(self) -> bool:
        return True

    # --- the bracket ----------------------------------------------------------------------
    def connect(self) -> None:
        self._bus.connect()

        # The bus raises on every normalise until a calibration is registered, so without this
        # present_position, get_observation, send_action, write_goal and joint_limits all fail.
        # Never overwrite a supplied one: read_calibration() hard-codes drive_mode to 0, and
        # zeroing a joint's drive_mode inverts its direction for the rest of the run.
        self._calibration_claimed = bool(self._bus.calibration)
        if not self._calibration_claimed:
            self._bus.calibration = self._bus.read_calibration()

        # get_observation() and send_action() are gated on every camera being open. On failure,
        # close whatever opened — a half-open environment must not be handed back.
        try:
            for cam in self._robot.cameras.values():
                cam.connect()
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        # Per camera and past every failure: this runs while unwinding from an arbitrary error, and
        # a camera that dropped out must not stop the bus being closed.
        for cam in self._robot.cameras.values():
            try:
                cam.disconnect()
            except Exception:  # noqa: BLE001,S110 - teardown, nothing left to salvage
                pass
        try:
            self._bus.disconnect()
        except Exception:  # noqa: BLE001,S110
            pass

    # --- the control loop -----------------------------------------------------------------
    def get_observation(self) -> dict:
        return self._robot.get_observation()

    def camera_sources(self) -> dict[str, str]:
        # Resolved in __init__, so this answers before connect().
        return dict(self._sources)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        # LeRobot re-reads the present position on every send and applies max_relative_target,
        # so the returned value is what the servos were actually told, not what was asked for.
        return self._robot.send_action(action)

    def present_position(self) -> dict[str, float]:
        return self._bus.sync_read("Present_Position")

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        # Reads nothing and changes nothing, so asking twice cannot move the envelope.
        #
        # `_normalize` is private LeRobot API, used deliberately: it is the same tick-to-degree
        # conversion `sync_read` applies to live positions, so limits and readings cannot disagree.
        # Restating the arithmetic here is what would let them. The client pins the LeRobot commit,
        # which is what makes depending on a private method safe enough.
        calibration = self._bus.calibration

        unusable = []
        for joint, cal in calibration.items():
            if cal.range_max <= cal.range_min:
                unusable.append(f"{joint}: range_max {cal.range_max} <= range_min {cal.range_min}")
            elif joint != "wrist_roll" and (cal.range_min, cal.range_max) == (0, self._full_span):
                # A full encoder span means the sweep never captured that joint. wrist_roll is
                # exempt: it rotates continuously, so the full span is its real range.
                unusable.append(
                    f"{joint}: a full 0..{self._full_span} range means its limits were never "
                    f"captured during calibration"
                )
        if unusable:
            raise ValueError("; ".join(unusable))

        ids = {cal.id: joint for joint, cal in calibration.items()}
        lo = self._bus._normalize({cal.id: cal.range_min for cal in calibration.values()})
        hi = self._bus._normalize({cal.id: cal.range_max for cal in calibration.values()})
        return {ids[i]: (min(lo[i], hi[i]), max(lo[i], hi[i])) for i in ids}

    def write_goal(self, pose: dict[str, float]) -> None:
        self._bus.sync_write("Goal_Position", pose)

    # --- limits the environment enforces on the caller's behalf ---------------------------
    def set_step_limit(self, per_joint: dict[str, float]) -> None:
        # Read by send_action on every call, against a freshly read present position.
        self._robot.config.max_relative_target = dict(per_joint)

    def set_motion_limits(self, acceleration: int, speed: int) -> None:
        # Per joint, because the bus has no broadcast write for these registers.
        for joint in self._bus.motors:
            self._bus.write("Acceleration", joint, acceleration, num_retry=2)
            self._bus.write("Goal_Velocity", joint, speed, num_retry=2)

    # --- queries --------------------------------------------------------------------------
    def temperatures(self) -> dict[str, int] | None:
        return self._bus.sync_read("Present_Temperature", normalize=False)

    def torque_state(self) -> dict[str, int] | None:
        return self._bus.sync_read("Torque_Enable", normalize=False)

    def is_calibrated(self) -> bool | None:
        # No claim, so the EEPROM is the frame and nothing independent exists to disagree with it.
        if not self._calibration_claimed:
            return None
        # LeRobot re-reads the EEPROM and compares it against the calibration it was configured
        # with. It compares ranges and homing offsets, not drive_mode, which is the other reason
        # connect() leaves a supplied calibration untouched.
        return self._bus.is_calibrated

    # --- unwinding ------------------------------------------------------------------------
    def disable_torque(self) -> None:
        self._bus.disable_torque(num_retry=2)

    def configure(self) -> None:
        self._robot.configure()
