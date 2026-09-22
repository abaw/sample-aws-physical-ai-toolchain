#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Score a GR00T N1.7 checkpoint offline and write the pipeline's gate input.
#   $1 checkpoint   $2 dataset root   $3 output dir   $4 traj-ids   $5 horizon   $6 max steps
#
# MUST write $3/metrics.json: the pipeline gates on that file and never reads logs. GR00T's own
# evaluator only PRINTS its numbers, so the scraping lives here — the pipeline must never learn
# another project's log format.
set -euo pipefail

CKPT_IN="${1:?score.sh needs a checkpoint}"
DS="${2:?score.sh needs a dataset root}"
OUT="${3:?score.sh needs an output directory}"
TRAJ_IDS="${4:?score.sh needs trajectory ids}"
HORIZON="${5:?score.sh needs an execution horizon}"
MAX_STEPS="${6:?score.sh needs a step cap}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="$(gr00t_resolve_checkpoint "${CKPT_IN}")"

echo "[gr00t-n17] score: trajs='${TRAJ_IDS}' horizon=${HORIZON} max_steps=${MAX_STEPS}"
mkdir -p "${OUT}/plots"

# Upstream's evaluator writes its per-trajectory plots to a fixed path of its own rather than
# anywhere it is told, so they are collected afterwards. Cleared first, or a re-run inside the
# same container would publish the previous run's plots alongside this one's.
PLOT_DIR=/tmp/open_loop_eval
rm -rf "${PLOT_DIR}"

# A checkpoint without normalisation statistics loads without complaint and then predicts at the
# wrong scale, which looks like a bad fine-tune. Name the real cause instead.
if [ ! -f "${CKPT}/experiment_cfg/dataset_statistics.json" ]; then
  echo "ERROR: ${CKPT} has no experiment_cfg/dataset_statistics.json — the training run did not publish a complete checkpoint" >&2
  exit 1
fi

cd "${GROOT_DIR}"
python gr00t/eval/open_loop_eval.py \
  --dataset-path "${DS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --model-path "${CKPT}" \
  --traj-ids ${TRAJ_IDS} \
  --execution-horizon "${HORIZON}" \
  --denoising-steps "${DENOISING_STEPS:-4}" \
  --steps "${MAX_STEPS}" 2>&1 | tee "${OUT}/openloop_eval.log"

# Smoothness, which the error metrics above cannot see: a zero-mean oscillation on a broadly
# correct trajectory scores well on MSE and MAE, and on a real arm it is what overheats a joint.
# Allowed to fail — a smoothness crash must not destroy a valid error measurement.
python -u "${HERE}/score_lib.py" "${DS}" "${CKPT}" \
  "${SMOOTHNESS_TRAJ_ID:-0}" "${SMOOTHNESS_STEPS:-160}" "${DENOISING_STEPS:-4}" \
  2>&1 | tee "${OUT}/smoothness.log" || echo "SMOOTHNESS_FAILED"
grep RESULTS_JSON "${OUT}/smoothness.log" | sed 's/^RESULTS_JSON //' > "${OUT}/smoothness.json" || true

# Copy whatever plots exist rather than letting an unmatched glob fail the task. The numbers are
# the gate; the plots are how a person sees WHERE it went wrong.
cp "${PLOT_DIR}"/*.jpeg "${OUT}/plots/" 2>/dev/null || true
echo "[gr00t-n17] plots collected: $(find "${OUT}/plots" -name '*.jpeg' | wc -l)"

# Assemble metrics.json. Deferred exit status: publish first, then fail, or a non-zero exit here
# would leave nothing to diagnose it from.
METRICS_RC=0
python3 - "${OUT}" "${TRAJ_IDS}" "${CHECKPOINT_PROVENANCE:-${CKPT}}" "${CKPT}" "${SMOOTHNESS_WARN_REVERSALS:-0.25}" <<'PY' || METRICS_RC=$?
import json, pathlib, re, sys

out = pathlib.Path(sys.argv[1])
requested = [int(t) for t in sys.argv[2].split()]
log = (out / "openloop_eval.log").read_text(errors="replace")

# The source URL, not the local path: inputs are downloaded to a numbered
# directory, so the path says nothing about which run or step was scored. The
# step comes from the checkpoint itself, and is absent from a final export.
state = pathlib.Path(sys.argv[4]) / "trainer_state.json"
global_step = None
if state.is_file():
    global_step = json.loads(state.read_text()).get("global_step")

def num(text):
    try:
        return float(text)
    except ValueError:
        return None

# Last occurrence wins, so a re-run of a trajectory within one log is not double
# counted.
per_traj = {}
for tid, mse, mae in re.findall(
    r"MSE for trajectory (\d+): (\S+?), MAE: (\S+)", log
):
    per_traj[int(tid)] = dict(traj_id=int(tid), mse=num(mse), mae=num(mae.rstrip(",")))

def avg(label):
    m = re.search(r"Average %s across all trajs: (\S+)" % label, log)
    return num(m.group(1)) if m else None

evaluated = sorted(t for t, v in per_traj.items() if v["mse"] is not None)
missing = [t for t in requested if t not in evaluated]
# Smoothness, if it was produced. Reported beside the error metrics because the two
# answer different questions and only one of them is visible in the plots.
smoothness = None
# Flat, contract-named figures, so one reader handles every model's metrics.json.
worst_reversals_predicted = None
worst_reversals_ground_truth = None
smooth_path = out / "smoothness.json"
if smooth_path.is_file() and smooth_path.stat().st_size:
    try:
        raw_smooth = json.loads(smooth_path.read_text())
        row = next(iter(raw_smooth["sweep"].values()))
        worst_reversals_predicted = row["worst_reversals"]
        worst_reversals_ground_truth = max(
            v["reversals"] for v in raw_smooth["ground_truth"].values()
        )
        smoothness = dict(
            worst_reversals=worst_reversals_predicted,
            ground_truth_worst_reversals=worst_reversals_ground_truth,
            per_joint={k: v["reversals"] for k, v in row["joints"].items()},
            travel_per_joint={k: v["travel"] for k, v in row["joints"].items()},
        )
    except (ValueError, KeyError, StopIteration) as exc:
        print("WARNING: could not read smoothness.json: %r" % exc, file=sys.stderr)

result = dict(
    checkpoint_url=sys.argv[3],
    global_step=global_step,
    requested_traj_ids=requested,
    evaluated_traj_ids=evaluated,
    missing_traj_ids=missing,
    per_trajectory=[per_traj[t] for t in evaluated],
    average_mse=avg("MSE"),
    average_mae=avg("MAE"),
    plot_count=len(list((out / "plots").glob("*.jpeg"))),
    smoothness=smoothness,
    complete=not missing and avg("MSE") is not None,
    # The contract's own spelling, so one gate reads every model's metrics.json.
    # Duplicated rather than renamed: average_mse and the nested smoothness object
    # are what this pipeline's own README and past runs report, and silently
    # renaming them would break every reader that already exists.
    mse=avg("MSE"),
    mae=avg("MAE"),
    worst_reversals_predicted=worst_reversals_predicted,
    worst_reversals_ground_truth=worst_reversals_ground_truth,
)
(out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

lines = [
    "checkpoint: %s (step %s)" % (result["checkpoint_url"], global_step),
    "trajectories requested: %s" % requested,
    "trajectories evaluated: %s" % evaluated,
]
lines += ["traj %d: MSE %s, MAE %s" % (t["traj_id"], t["mse"], t["mae"])
          for t in result["per_trajectory"]]
lines += ["average MSE: %s" % result["average_mse"],
          "average MAE: %s" % result["average_mae"]]
if smoothness:
    lines.append(
        "worst-joint direction reversals: %.1f%% (ground truth %.1f%%)"
        % (smoothness["worst_reversals"] * 100,
           smoothness["ground_truth_worst_reversals"] * 100)
    )
(out / "metrics.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# Loud, and deliberately not a failure: a jittery policy is still drivable, but only
# with stage 04's command deadband. Silence here is what let it reach hardware.
if smoothness and smoothness["worst_reversals"] > float(sys.argv[5]):
    print(
        "WARNING: this checkpoint's action chunks reverse direction on %.0f%% of "
        "steps at their worst joint, against %.1f%% for the trajectory it imitates. "
        "The arm will do servo work without moving. Drive it with stage 04's "
        "command_deadband_deg set, and watch servo temperature."
        % (smoothness["worst_reversals"] * 100,
           smoothness["ground_truth_worst_reversals"] * 100),
        file=sys.stderr,
    )

if not result["complete"]:
    print("INCOMPLETE: requested %s, evaluated %s — trajectory IDs out of range "
          "for this dataset, or the eval reported no averages"
          % (requested, evaluated), file=sys.stderr)
    sys.exit(1)
PY

if [ "${METRICS_RC}" -ne 0 ]; then
  echo "[gr00t-n17] metrics.json reports a failure; see above" >&2
  exit "${METRICS_RC}"
fi
echo "[gr00t-n17] score done"
