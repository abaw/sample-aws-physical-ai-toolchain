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
"""Model-agnostic checks on an SO-101 LeRobot dataset.

Deliberately stdlib + ffmpeg only — no torch, no model toolchain — so a dataset can be
validated for a LeRobot-native policy without a multi-GB install.

Env: DS (dataset root), EXPECT_EPISODES (0 to skip), EXPECT_STATE_DIM.
Exits non-zero if any check fails.
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

DS = Path(os.environ["DS"])
EXPECT_EPISODES = int(os.environ.get("EXPECT_EPISODES", "0"))
EXPECT_STATE_DIM = int(os.environ.get("EXPECT_STATE_DIM", "6"))

failures = []


def check(ok, label, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print("== layout ==")
info = json.loads((DS / "meta" / "info.json").read_text())
parquet = glob.glob(str(DS / "data" / "**" / "*.parquet"), recursive=True)
videos = sorted(glob.glob(str(DS / "videos" / "**" / "*.mp4"), recursive=True))
cams = sorted(k for k in info["features"] if k.startswith("observation.images."))

# v2 and v3 are both legitimate inputs: v3 is what LeRobot records today and what a
# LeRobot-native policy reads directly. Only GR00T requires v2, so that assertion lives
# in the per-model checks.
version = info["codebase_version"]
check(version.startswith(("v2", "v3")), "codebase_version is v2.x or v3.x", version)
is_v3 = version.startswith("v3")
check((DS / "meta" / "info.json").exists(), "meta/info.json present")

# The two versions store episode and task metadata differently, and pack the data and
# video trees differently, so the layout checks have to branch. v2 writes one file per
# episode; v3 packs many episodes into chunked files, so per-episode counting does not
# apply to it and only presence can be asserted.
if is_v3:
    check((DS / "meta" / "tasks.parquet").exists(), "meta/tasks.parquet present")
    check(
        bool(glob.glob(str(DS / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)),
        "meta/episodes/ parquet present",
    )
    check(len(parquet) > 0, "data parquet present", f"{len(parquet)} files")
    for cam in cams:
        check(
            bool(glob.glob(str(DS / "videos" / cam / "**" / "*.mp4"), recursive=True)),
            f"videos for {cam} present",
        )
else:
    for name in ("episodes.jsonl", "tasks.jsonl"):
        check((DS / "meta" / name).exists(), f"meta/{name} present")
    check(
        len(parquet) == info["total_episodes"],
        "one parquet per episode",
        f"{len(parquet)} files vs total_episodes {info['total_episodes']}",
    )
    check(
        len(videos) == info["total_episodes"] * len(cams),
        "one video per episode per camera",
        f"{len(videos)} files vs {info['total_episodes']} x {len(cams)}",
    )
    # Advisory field, absent from v3. Recordings exist with it wrong and no loader reads
    # it, so warn rather than fail — and do not rewrite metadata nothing consumes.
    if info.get("total_videos") != len(videos):
        print(
            f"warn info.json total_videos={info.get('total_videos')} but {len(videos)} "
            "video files exist (advisory field; loaders do not read it)"
        )
check(
    info["features"]["observation.state"]["shape"] == [EXPECT_STATE_DIM],
    f"state dim is {EXPECT_STATE_DIM}",
    str(info["features"]["observation.state"]["shape"]),
)
check(
    info["features"]["action"]["shape"] == [EXPECT_STATE_DIM],
    f"action dim is {EXPECT_STATE_DIM}",
    str(info["features"]["action"]["shape"]),
)
if EXPECT_EPISODES:
    check(
        info["total_episodes"] == EXPECT_EPISODES,
        f"episode count is {EXPECT_EPISODES}",
        str(info["total_episodes"]),
    )
print(
    f"info: {version}, {info['total_episodes']} episodes, {info['total_frames']} frames, "
    f"{info['fps']} fps, cameras {cams}"
)
# Printed, not asserted: the order must match the policy's modality config and the eval
# client, which is checked per-model.
print(f"joint names: {info['features']['observation.state']['names']}")

# LeRobot records AV1 by default, and a decoder is only present if ffmpeg is. Getting
# this wrong surfaces as a crash hours into training, so decode here.
print("== video decode ==")
probe = subprocess.run(
    [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_frames",
        "-of", "json", videos[0],
    ],
    capture_output=True,
    text=True,
)
meta = json.loads(probe.stdout)["streams"][0] if probe.returncode == 0 else {}
print(f"{videos[0]}: {meta}")
check(bool(meta.get("codec_name")), "ffprobe read the stream", probe.stderr[:200])

# ffprobe reads container metadata even for a codec it cannot decode, so decode a frame
# to raw and check the byte count.
raw = "/tmp/frame.rawvideo"
dec = subprocess.run(
    [
        "ffmpeg", "-v", "error", "-y", "-i", videos[0],
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", raw,
    ],
    capture_output=True,
    text=True,
)
expected_bytes = int(meta.get("width", 0)) * int(meta.get("height", 0)) * 3
actual = os.path.getsize(raw) if os.path.exists(raw) else 0
check(
    dec.returncode == 0 and actual == expected_bytes and expected_bytes > 0,
    f"decoded one {meta.get('codec_name')} frame",
    f"{actual} bytes, expected {expected_bytes}; {dec.stderr[:200]}",
)
# A decoder that silently emits a blank frame is worse than one that errors.
if actual:
    data = Path(raw).read_bytes()
    spread = max(data) - min(data)
    check(spread > 8, "decoded frame is not blank", f"value spread {spread}")

print()
if failures:
    print(f"COMMON VALIDATION FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("COMMON VALIDATION PASSED")
