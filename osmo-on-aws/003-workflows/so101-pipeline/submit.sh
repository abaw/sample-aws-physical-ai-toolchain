#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Submit the pipeline without hand-assembling --set arguments.
#
#   ./submit.sh --example full-chain --model gr00t-n17 \
#       storage_url=s3://<bucket>/so101 \
#       source_dataset_url=s3://<bucket>/raw/<name>/ \
#       dataset_name=<name> \
#       traj_ids="3 7 11" \
#       lang_instruction="Lift the red cube and put to the black plate"
#
#   ./submit.sh --example hil-only --model pi0fast --dry-run ...
#
# Everything is key=value. The script works out the rest:
#
#   * WHICH FLAG. `--set` coerces values to int/float where it can, which the spec depends on --
#     a storage figure is computed arithmetically from several variables, and a string there breaks
#     the render. Everything else goes to `--set-string`, so a value that merely looks numeric at
#     the front, like a trajectory list, is not silently truncated to its first number.
#   * QUOTING. Values reach the script as ordinary shell arguments, so spaces and globs are safe.
#     Hand-assembling `--set $(cat file)` word-splits, and a camera selector's `*` is expanded
#     against the working directory before OSMO ever sees it.
#   * TYPOS. `--set` accepts keys nothing references, SILENTLY, so a mistyped flag does nothing and
#     the run proceeds on a default. This checks every key against the spec's own default-values
#     and refuses an unknown one. That is the single trap the platform cannot catch for you.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SPEC="${HERE}/pipeline.yaml"
EXAMPLE=""
MODEL=""
PASSTHROUGH=()
CLI=()
declare -a ORDER=()
declare -A VALUES=()

die() { echo "ERROR: $*" >&2; exit 2; }

put() {
  local key="${1%%=*}" value="${1#*=}"
  [ "${key}" = "$1" ] && die "expected key=value, got '$1'"
  [ -z "${key}" ] && die "empty key in '$1'"
  # A later assignment wins, and the key keeps its first position so the printed command reads in
  # the order a user wrote it.
  [ -n "${VALUES[$key]+set}" ] || ORDER+=("${key}")
  VALUES["${key}"]="${value}"
}

load_vars() {
  local file="$1"
  [ -f "${file}" ] || die "no such file: ${file}"
  local line
  while IFS= read -r line || [ -n "${line}" ]; do
    line="${line%%#*}"
    line="$(printf '%s' "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -z "${line}" ] && continue
    put "${line}"
  done < "${file}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --example) EXAMPLE="${2:?--example needs a name}"; shift 2 ;;
    --model)   MODEL="${2:?--model needs a name}"; shift 2 ;;
    --spec)    SPEC="${2:?--spec needs a path}"; shift 2 ;;
    --dry-run|--validate) PASSTHROUGH+=("$1"); shift ;;
    -h|--help) sed -n '3,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *=*)       CLI+=("$1"); shift ;;
    *)         die "unrecognised argument '$1' (everything else must be key=value)" ;;
  esac
done

# Example first, then the model's own variables, then the command line — so a user's explicit value
# always wins over a file's.
if [ -n "${EXAMPLE}" ]; then
  CANDIDATE="${HERE}/examples/${EXAMPLE}.vars"
  [ -f "${CANDIDATE}" ] || die "no example '${EXAMPLE}'. Available: $(cd "${HERE}/examples" && ls *.vars | sed 's/\.vars$//' | paste -sd, -)"
  load_vars "${CANDIDATE}"
fi
if [ -n "${MODEL}" ]; then
  [ -d "${HERE}/models/${MODEL}" ] || die "no model '${MODEL}'. Available: $(cd "${HERE}/models" && ls -d */ | tr -d / | paste -sd, -)"
  put "model=${MODEL}"
  [ -f "${HERE}/models/${MODEL}/model.vars" ] && load_vars "${HERE}/models/${MODEL}/model.vars"
fi

# The command line goes on LAST, so an explicit value always beats a file's. Collected during
# parsing rather than applied there, because the files are read after the flags are known.
for assignment in ${CLI[@]+"${CLI[@]}"}; do put "${assignment}"; done

[ "${#ORDER[@]}" -eq 0 ] && die "nothing to submit. Try --example full-chain --model gr00t-n17 ..."

# Every key must exist in the spec, or it does nothing and says nothing.
mapfile -t KNOWN < <(awk '/^default-values:/{f=1;next} /^[^[:space:]#]/{f=0} f && /^  [a-z_]+:/{sub(/:.*/,"");gsub(/ /,"");print}' "${SPEC}")
UNKNOWN=()
for key in "${ORDER[@]}"; do
  found=0
  for k in "${KNOWN[@]}"; do [ "${k}" = "${key}" ] && { found=1; break; }; done
  [ "${found}" -eq 0 ] && UNKNOWN+=("${key}")
done
if [ "${#UNKNOWN[@]}" -gt 0 ]; then
  echo "ERROR: these keys are not declared in ${SPEC##*/}, so --set would ignore them silently:" >&2
  for k in "${UNKNOWN[@]}"; do echo "         ${k}" >&2; done
  echo "       Declared keys: ${KNOWN[*]}" >&2
  exit 2
fi

# Numeric values keep --set so the spec's arithmetic still works; everything else is forced to a
# string so a leading number cannot swallow the rest of the value.
SET_ARGS=()
STR_ARGS=()
for key in "${ORDER[@]}"; do
  value="${VALUES[$key]}"
  if [[ "${value}" =~ ^-?[0-9]+$ || "${value}" =~ ^-?[0-9]*\.[0-9]+$ ]]; then
    SET_ARGS+=("${key}=${value}")
  else
    STR_ARGS+=("${key}=${value}")
  fi
done

CMD=(osmo workflow submit "${SPEC}")
[ "${#PASSTHROUGH[@]}" -gt 0 ] && CMD+=("${PASSTHROUGH[@]}")
[ "${#SET_ARGS[@]}" -gt 0 ] && CMD+=(--set "${SET_ARGS[@]}")
[ "${#STR_ARGS[@]}" -gt 0 ] && CMD+=(--set-string "${STR_ARGS[@]}")

# Printed before running, so the run is auditable and the command can be copied if the helper is
# ever in the way.
echo "+ $(printf '%q ' "${CMD[@]}")" >&2
exec "${CMD[@]}"
