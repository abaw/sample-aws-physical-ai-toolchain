#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Port a LeRobot recording to the v3.0 layout LeRobot's smolvla policy reads.
#   $1  input dataset root (read-only)     $2  output directory
#
# This model's format requirement is the exact opposite of the other shipped one: it needs v3.0
# and raises BackwardCompatibilityError on anything older, while that one reads v2 and converts
# v3 down. Neither can read the other's tree, which is why conversion is per-model and its output
# prefix is keyed by model.
#
# Idempotent: a tree that is already v3.0 is published unchanged, so re-running is safe.
set -euo pipefail

IN="${1:?prepare.sh needs an input dataset root}"
OUT="${2:?prepare.sh needs an output directory}"
HERE="$(cd "$(dirname "$0")" && pwd)"

INFO="$(find "${IN}" -name info.json -path '*/meta/*' -print -quit)"
if [ -z "${INFO}" ]; then
  echo "ERROR: no meta/info.json under ${IN} — is the source a LeRobot dataset root?" >&2
  find "${IN}" -maxdepth 3 >&2
  exit 1
fi
SRC="$(dirname "$(dirname "${INFO}")")"
SRC_VERSION="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['codebase_version'])" "${SRC}/meta/info.json")"
echo "[smolvla] source: ${SRC} (codebase_version=${SRC_VERSION})"

# Convert in scratch, then copy the result out. NOT in the output directory itself: LeRobot's
# converter renames its target to <root>_old, and $2 is a mount point, so renaming it fails with
# "Device or resource busy" — after the conversion work is already done. And not in the input
# either: the contract calls $1 read-only.
DS=/tmp/prepare_work
rm -rf "${DS}" "${DS}_old" "${DS}_v30"
mkdir -p "${DS}"
cp -a "${SRC}/." "${DS}/"

if [ "${SRC_VERSION}" = "v3.0" ]; then
  echo "[smolvla] already v3.0 — publishing unchanged"
else
  # Datasets written by other tooling declare v2.1 while carrying the v2.0 metadata layout: no
  # meta/episodes_stats.jsonl. LeRobot's converter validates the version string, believes it,
  # and dies on the missing file. The v2.0 -> v2.1 script that used to write it was removed in
  # v0.6.x, hence this backfill.
  python "${HERE}/prepare_lib.py" "${DS}"

  # --root is the exact folder holding meta/, data/, videos/. The converter builds <root>_v30,
  # moves it over <root> and leaves the original at <root>_old, so afterwards ${DS} is the v3
  # tree. --push-to-hub=false keeps it local.
  python "${LEROBOT_DIR}/src/lerobot/scripts/convert_dataset_v21_to_v30.py" \
    --repo-id="${REPO_ID:-local/so101-smolvla}" \
    --root="${DS}" \
    --push-to-hub=false

  # Episode and frame counts are what catch a partial conversion; a version bump alone would not.
  python3 - "${SRC}/meta/info.json" "${DS}/meta/info.json" <<'PY'
import json, sys
before, after = (json.load(open(p)) for p in sys.argv[1:3])
print(f"converted: codebase_version={after['codebase_version']} "
      f"episodes={after.get('total_episodes')} frames={after.get('total_frames')}")
if not str(after["codebase_version"]).startswith("v3"):
    sys.exit(f"ERROR: conversion left codebase_version={after['codebase_version']}")
for key in ("total_episodes", "total_frames"):
    if before.get(key) != after.get(key):
        sys.exit(f"ERROR: {key} changed in conversion: {before.get(key)} -> {after.get(key)}")
print("episode and frame counts preserved")
PY

  # The converter leaves the pre-conversion tree beside the result. Copying it out would double
  # the upload and hand the next stage two trees to choose between.
  rm -rf "${DS}_old" "${DS}_v30"
fi

cp -a "${DS}/." "${OUT}/"
rm -rf "${DS}"

echo "[smolvla] prepared $(find "${OUT}" -type f | wc -l) files"
