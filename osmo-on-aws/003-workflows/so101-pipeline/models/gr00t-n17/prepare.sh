#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Convert a LeRobot recording into the flavour GR00T N1.7 reads.
#   $1  input dataset root (read-only)
#   $2  output directory — publish the converted tree here
#
# GR00T wants v2 plus a meta/modality.json that a stock LeRobot dataset does not carry. The
# version is read off the dataset rather than taken as an argument: a mismatched flag would
# either mangle an already-converted tree or train on the wrong layout, and both are silent.
#
# Idempotent, as the contract requires — the prepared prefix is keyed by dataset NAME so it is
# a cache, and re-running must be safe. Re-running on a v2 tree is a no-op beyond the copy.
set -euo pipefail

IN="${1:?prepare.sh needs an input dataset root}"
OUT="${2:?prepare.sh needs an output directory}"

# Find the root by the file that must exist rather than assuming nesting depth: OSMO downloads
# an input into a numbered directory whose internal layout follows whatever was uploaded.
INFO="$(find "${IN}" -name info.json -path '*/meta/*' -print -quit)"
if [ -z "${INFO}" ]; then
  echo "ERROR: no meta/info.json under ${IN} — is the source a LeRobot dataset root?" >&2
  find "${IN}" -maxdepth 3 >&2
  exit 1
fi
SRC="$(dirname "$(dirname "${INFO}")")"
NAME="$(basename "${SRC}")"
echo "[gr00t-n17] source: ${SRC}"

# Convert in the OUTPUT tree, never in the input: the contract calls $1 read-only, and the v3
# converter writes alongside its source, so doing this in place would mutate a shared download.
WORK="${OUT}/${NAME}"
mkdir -p "${WORK}"
cp -a "${SRC}/." "${WORK}/"

DS_VERSION="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['codebase_version'])" "${WORK}/meta/info.json")"
echo "[gr00t-n17] codebase_version: ${DS_VERSION}"

case "${DS_VERSION}" in
  v3*)
    echo "[gr00t-n17] downgrading ${DS_VERSION} -> v2.1"
    ( cd "${GROOT_DIR}" && uv run --project scripts/lerobot_conversion \
        python scripts/lerobot_conversion/convert_v3_to_v2.py \
        --repo-id "${NAME}" --root "${OUT}" )
    ;;
  v2*) echo "[gr00t-n17] already ${DS_VERSION}; no downgrade needed" ;;
  *)   echo "ERROR: unsupported codebase_version ${DS_VERSION}" >&2; exit 1 ;;
esac

# modality.json is GR00T-specific and absent from a stock LeRobot dataset. It maps this model's
# configured camera keys onto the feature names the recording actually carries, and refuses if
# the recording does not have them -- a mapping that disagrees with the data trains to
# convergence and then drives the arm on another camera's frames.
python "${MODEL_FILES}/write_modality.py" "${WORK}"

# GR00T-specific checks: the modality mapping against this model's registered config. A
# mismatch here is silent at training time and surfaces as an arm moving the wrong joint.
cd "${GROOT_DIR}"
DS="${WORK}" python "${MODEL_FILES}/check_dataset.py"

# Flatten: the pipeline publishes $2 itself, so the tree has to sit at its root. `mv` rather
# than a second copy — the work tree has no further use and these are large directories.
cd - >/dev/null
for d in data videos meta; do
  [ -e "${OUT}/${d}" ] && rm -rf "${OUT}/${d}"
  mv "${WORK}/${d}" "${OUT}/"
done
rm -rf "${WORK}"

echo "[gr00t-n17] prepared $(find "${OUT}" -type f | wc -l) files"
