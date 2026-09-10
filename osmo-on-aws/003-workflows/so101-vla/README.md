# SO-101 VLA fine-tuning on OSMO

Fine-tune a vision-language-action model on your own SO-101 robot recordings, judge the result before
it ever drives hardware, and then close the loop on the real arm — four OSMO workflows, one GPU.

The model is [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T). The dataset is whatever your SO-101
recorded, in [LeRobot](https://github.com/huggingface/lerobot) format.

| Stage | Workflow | GPU | Measured wall clock | Publishes |
|---|---|:---:|---|---|
| 01 | `01_prepare_dataset.yaml` | none needed | ~4 min | a validated dataset at a stable prefix |
| 02 | `02_finetune.yaml` | 1 | ~10 min for a 10-step smoke run; **~6 h for 10k steps** | checkpoints, per run |
| 03 | `03_openloop_eval.yaml` | 1 | ~9 min | per-dimension plots and error metrics |
| 04 | `04_hil_eval.yaml` | 1 | ~20 min, most of it loading the checkpoint | closed-loop results, trace, and latency |

Run them in order. Each reads what the previous published, so the seam between them is an S3 prefix
rather than a shared filesystem — any stage can be re-run alone.

**Stages 01-03 need no robot.** They are a complete cloud pipeline, and they are useful on their own:
stage 03 is the regression gate you want *before* touching hardware. **Stage 04 needs a physical
SO-101 arm on an EKS Hybrid Node** and is the only stage that moves anything.

**Stage 03 is the point of the exercise.** An under-trained policy, a wrong joint order, and missing
normalization statistics all look identical on real hardware — an arm that moves wrongly — and stage 03
tells them apart for the price of a few GPU-minutes. Treat it as a stop/go gate, not an optional extra.

## Prerequisites

**A pool with a GPU platform.** All three stages default to `platform: l40s`, the platform this repo's
setup creates; override per stage with `--set prep_platform=`, `train_platform=`, `eval_platform=`.
Stage 01 needs no GPU but requests the GPU platform anyway, because a platform that sets `USER_GPU` in
its `default_variables` grants a GPU regardless — OSMO reads `gpu: 0` as absent, not as an override.
Point it at a CPU platform if your pool has one with ≥ 2 spare vCPU.

**A Hugging Face token, as an OSMO credential:**

```bash
osmo credential set huggingface_token --type GENERIC --payload-file token=<file-holding-the-token>
```

This is not optional for stages 02 and 03. The base model is public, but it ships no tokenizer and pulls
its vision-language backbone and processor from a separate **gated** repository at load time, so an
anonymous run dies during model construction. The Hub account behind the token must have **accepted that
repository's terms** — a token with the right scope still gets HTTP 403 until someone clicks through.

**An S3 bucket** the workflow role can read and write. Everything below writes under one prefix you
choose.

**A dataset** in LeRobot format, v2 or v3, either in S3 or on the Hub. Stage 01 detects which.

## Before your first submit: how `--set` works

`osmo workflow submit` takes **every** override in a single `--set`:

```bash
--set storage_url=s3://<bucket>/so101-vla dataset_url=s3://<bucket>/so101-vla/raw/<name>/
```

Repeating the flag is **not** an error and is **not** additive — only the last group is applied, and the
run proceeds on the placeholder defaults in the workflow. Because OSMO downloads a task's inputs before
the task's own script runs, the workflow cannot catch this for you: it fails at input download with
`NoSuchBucket` against a bucket named `REPLACE-WITH-YOUR-DATASETS-BUCKET`.

`osmo workflow submit <file> --dry-run` prints the rendered spec and is the fastest way to confirm your
overrides landed. `osmo workflow validate <file>` checks it server-side without consuming a node.

## Running the pipeline

Throughout, `B` is the prefix everything lives under:

```bash
B=s3://<bucket>/so101-vla
```

### 1. Prepare and validate the dataset

```bash
osmo workflow submit 003-workflows/so101-vla/01_prepare_dataset.yaml \
  --set storage_url=$B \
        dataset_url=$B/raw/<name>/ \
        dataset_name=<name> \
        dataset_size_gi=<size-of-your-dataset-rounded-up>
```

From the Hub instead: `--set dataset_source=hf dataset_repo_id=<owner>/<repo> …`.

Most of this stage's value is in what it refuses. It checks episode count and state dimensionality
against what you assert, converts a v3 recording to the v2 layout the model needs, and ensures the
modality mapping exists. A dataset that trains for hours and then drives the wrong joint is the failure
it exists to prevent.

For a LeRobot-native policy, `--set model=none` validates and publishes without installing a GR00T
toolchain it will never use.

### 2. Fine-tune

**Smoke-test first.** Ten steps exercises the whole path — model download, dataset load, optimizer step,
checkpoint upload — in about ten minutes instead of hours, and leaves a real checkpoint for stage 03:

```bash
osmo workflow submit 003-workflows/so101-vla/02_finetune.yaml \
  --set storage_url=$B \
        dataset_url=$B/dataset/<name>/ \
        dataset_size_gi=<size> \
        max_steps=10 save_steps=10 exec_timeout=1h
```

Then the real run, with `max_steps` in the 10k-20k range — about 6 h for 10k steps. Do not budget
`exec_timeout` from a smoke run's step rate: a short run is dominated by start-up and reports roughly
twice the throughput a sustained run achieves. **A task killed by its timeout publishes nothing.**

Checkpoints land under `$B/checkpoints/<model>/<workflow-id>/`, one prefix per run, so two experiments
cannot overwrite each other and later stages can pin an exact run.

Every published directory carries a `manifest.json` of its own files and their sizes. This exists because
OSMO's uploader logs per-file failures and then reports success anyway, and the training task cannot check
its own upload — the uploader runs after the task container exits. The manifest travels with the
checkpoint, so stage 03 can tell a complete checkpoint from a truncated one before evaluating it.

### 3. Evaluate, before any hardware

```bash
osmo workflow submit 003-workflows/so101-vla/03_openloop_eval.yaml \
  --set storage_url=$B \
        dataset_url=$B/dataset/<name>/ \
        checkpoint_url=$B/checkpoints/gr00t-n17/<workflow-id>/checkpoint-<step>/ \
        traj_ids="3 7 11"
```

Name **one** `checkpoint-<step>/` prefix. Pointing `checkpoint_url` at a whole run prefix works, but a
run retains five checkpoints of ~24 GiB each and all of them are downloaded before the task starts.

Use several trajectories — one lucky trajectory is not convergence.

### 4. Close the loop on the real arm

Only if you have one. This stage physically actuates a robot — read
[Hardware-in-the-loop](#hardware-in-the-loop-stage-04) before submitting it, and rehearse with
`no_motion=true` first.

**Hold episodes back yourself if you want a generalisation measure.** Stage 02 trains on *every* episode
in the dataset it is given, so evaluating the resulting checkpoint against that same dataset measures how
well the policy fits data it has already seen. That is still worth doing — it catches joint-order and
normalization faults, and a policy that cannot even reproduce its training data is not worth putting near
an arm — but it will not tell you whether the policy generalises. For that, run stage 01 twice to publish
a training prefix and a holdout prefix from disjoint episodes, then point stage 02 at the first and
stage 03 at the second.

## Reading the result

Stage 03 publishes one plot per trajectory, `metrics.json`, `metrics.txt` and the full run log.

**Gate on `metrics.json`, not on the task status alone.** The task succeeding means the eval ran; it does
not mean the eval covered what you asked for. Trajectory IDs past the end of the dataset are skipped
with a warning and the eval still exits successfully, so a typo silently shrinks the sample the average
is computed over. `metrics.json` records `requested_traj_ids`, `evaluated_traj_ids` and a `complete`
flag for exactly that, and the task fails — after publishing everything — when they disagree:

```json
{ "checkpoint_url": "s3://<bucket>/so101-vla/checkpoints/gr00t-n17/<run-id>/checkpoint-10/",
  "global_step": 10, "requested_traj_ids": [0, 1, 2],
  "evaluated_traj_ids": [0, 1, 2], "missing_traj_ids": [],
  "average_mse": 1344.6, "average_mae": 30.0, "plot_count": 3, "complete": true }
```

The checkpoint is identified by its source URL and the step read from the checkpoint itself, not by a
local path: inputs are downloaded into a numbered directory, which says nothing about which run produced
them.

Upstream's
["Interpreting the Result: Is My Fine-tune Working?"](https://github.com/NVIDIA/Isaac-GR00T) section is
the rubric; read the plots against it rather than inventing one.

What the plots separate:

- **Ground truth tracked by the inference points, prediction is noise** → the policy is under-trained.
  Modality mapping and normalization statistics are *correct*; train longer.
- **Prediction tracks the wrong channel** → a joint-order or modality mistake in stage 01.
- **Everything at the wrong scale** → missing normalization statistics. Both stages fail the run rather
  than publish a checkpoint without them, so this should not reach you.

For calibration, measured on this example's 43-episode dataset at four trajectories each:

| Checkpoint | Average MSE | Average MAE |
|---|---|---|
| 10 steps (a smoke run) | 1347 | 30.0 |
| 6,000 steps | 15.5 | 2.44 |
| 10,000 steps | 9.06 | 1.55 |

The 10-step row is the floor: if a checkpoint that has barely trained ever looks *good*, suspect the
evaluation rather than the model. Two orders of magnitude separate it from a trained run, so this metric
discriminates strongly — and error still fell between 6,000 and 10,000 steps on every trajectory, so
10,000 is not past the useful point for a dataset this size. These are training episodes, so read them as
fit, not generalisation.

`execution_horizon` and `denoising_steps` are exposed here because they are also the two knobs that set
a real-time control loop's latency budget. Evaluate at the values you intend to drive an arm with.

## Hardware-in-the-loop (stage 04)

Stage 04 is one gang-scheduled group of two tasks: a **policy server** on a GPU node, and a **robot
client** on the machine the arm is plugged into. They are one group, not two workflows, because the
robot cannot usefully start before the server and a half-started group would leave the arm connected
with nothing driving it. The client carries `lead: true`, so when it finishes the group ends and the
GPU task is torn down with it.

### Safety comes first, and it is not optional

**A person must be present, with a way to cut power, for every run that can move the arm.** Everything
below reduces the size of a mistake; none of it removes the need for that.

What the client does, and why each item is there:

| Guard | What it prevents |
|---|---|
| A command deadband, and a servo-temperature rest | This policy does not hold still — see **Jitter** below. Without the deadband the arm pays acceleration current every step without moving, and the smallest servo reaches its thermal cutoff, at which it drops the arm |
| Torque is enabled only after `Goal_Position` is set to the *present* pose | LeRobot's own `connect()` re-enables torque onto whatever goal was last written — so an arm moved by hand since the last run would snap back to it |
| Every commanded value clamped to the calibrated range, less a margin | The calibrated range is narrower than the servo range, so an out-of-range action would drive into a mechanical hard stop |
| Per-step cap on how far a joint may be commanded from where it actually is | Bounds a single wild action to a slow move instead of a snap |
| Servo acceleration and speed capped well below LeRobot's teleoperation defaults | A bad target arrives slowly enough to react to |
| Return-to-start and torque release on every exit path, plus `SIGTERM`/`SIGINT`/`SIGHUP` handlers | **The real hazard is not a wrong action — it is the task dying while the arm holds a pose.** A cancelled workflow, a drained node or a lost tunnel must all end with the arm limp |
| Torque-off is **read back off the servos** and retried until confirmed | A Feetech write can fail transiently, and a servo that browns out mid-sequence returns with its EEPROM default — which on this arm is torque *enabled*. An unverified release is not a release |
| Torque-off is attempted whether or not the engage succeeded | A part-way engage leaves torque indeterminate, which is precisely when release must not be skipped |
| The engage sequence is retried, forcing torque off between attempts | The bus drops a packet often enough to see it; a half-applied engage must not become a held pose |
| The return ramp **waits for the arm to arrive** before dropping torque | Sending the last goal is not reaching it. With speed capped for safety the arm is still travelling, so releasing immediately leaves it wherever it got to |
| The servos' own EEPROM is asserted equal to the recording calibration | A policy trained in one calibrated joint frame and executed in another drives confidently to the wrong pose, with a workflow that reports success |
| Our unit conversion is checked against LeRobot's on the live arm before torque | Same failure, one level lower: a units bug here is silent |
| The whole policy path is exercised — observation, server validation, chunk length, action range — *before* torque | If any of it is wrong, the run ends with the arm untouched |

**The one gap, stated plainly: `SIGKILL` cannot be caught.** An OOM kill or a lost node leaves the
servos holding their last goal, because a Feetech bus has no watchdog. That is what the memory
headroom, the gripper torque limits, and a reachable physical e-stop are for. Do not rely on software
to save you from this one.

### Rehearse in this order

Each rung proves something the next one depends on, and only the last can move the arm.

1. **Stage 03 on this exact checkpoint.** Do not skip it. An under-trained policy looks like a
   miswired one on hardware.
2. **`--set no_motion=true`** with the arm powered but limp. This runs *everything* — both tasks
   schedule, the client resolves the policy task's host and connects, the observation passes the
   server's validation, the chunk length is checked, real end-to-end latency is measured, and every
   action the policy asks for is logged and range-checked — while performing **zero writes** to the
   servo bus. Every write path in the client is gated behind this flag.
3. **Read `results.json` from that run.** In particular `warmup_actions_outside_envelope` and
   `clipped_by_joint`: if the policy is asking for poses the arm cannot reach, find out here.
4. **One episode, powered, workspace clear, hand on the power.** `max_episodes=1`, a short
   `episode_timeout_s`, and stay next to it.

```bash
# rung 2 — cannot move the arm
osmo workflow submit 003-workflows/so101-vla/04_hil_eval.yaml \
  --set storage_url=$B \
        checkpoint_url=$B/checkpoints/gr00t-n17/<workflow-id>/checkpoint-<step>/ \
        lang_instruction="<the text your dataset was annotated with>" \
        no_motion=true max_episodes=1 episode_timeout_s=20
```

Drop `no_motion=true` for rung 4. Nothing else needs to change.

### Aborting and scoring, from outside the container

A container has no keyboard, so the client watches for sentinel files and `osmo workflow exec` can
reach in and create them:

```bash
osmo workflow exec <workflow-id> robot-client -- touch /tmp/hil-control/abort
osmo workflow exec <workflow-id> robot-client -- touch /tmp/hil-control/success
```

`abort` halts after the current step, returns the arm and releases it. `success` and `fail` end the
current episode with that verdict — **success is operator-scored, because there is no sensor for
"the cube is on the plate"**. An episode that simply runs out of time is recorded as `timeout` with
`success: null`, which is honest rather than a guess.

### The latency budget: measure it on your own link, and read this first

One chunk buys `action_horizon / control_hz` of motion. At 30 Hz, horizon 8 is **267 ms** and horizon
16 is **533 ms**. If a round trip costs more than that, the arm finishes every chunk before the next
one arrives and pauses at each boundary. The client works this out at warm-up and says so, with the
remedy, before torque is enabled.

**Component measurements will mislead you here, and that is the trap in this stage.** Adding up
61 ms of node-to-node ICMP, ~9 ms of serialization and 146 ms of in-process inference gives 216 ms and
predicts that horizon 8 fits comfortably. Measured end to end over the real pod-to-pod path, it does
not:

| Cameras | Horizon | Payload / request | p50 | p95 | p99 | max | Achieved rate | Timeouts |
|---|---|---|---|---|---|---|---|---|
| 2 × 640×480 | 8 | 1.84 MB | 380 ms | 705 ms | 893 ms | 1018 ms | 11.7 Hz | 0 (3000 ms ceiling) |
| 2 × 320×240 | 8 | 0.46 MB | **377 ms** | 584 ms | 717 ms | 794 ms | 12.4 Hz | 1 (1000 ms ceiling) |
| 2 × 640×480 | **16** | 1.84 MB | 518 ms | 791 ms | 1013 ms | 1084 ms | **15.2 Hz** | 0 (1500 ms ceiling) |

Two things to take from that.

**Cutting the payload by 4× moved the median by 3 ms.** So the round trip is not throughput-bound, and
shrinking the frames is not the lever it looks like — it buys a tighter *tail* (p95 705 → 584 ms) and
nothing at the median. The ~230 ms that component measurements do not account for is the real
pod-to-pod path: two overlay hops, Cilium's VXLAN and then the tunnel, plus the ZMQ round trip. A ping
between *nodes* does not measure it.

**The median itself moved 377 → 518 ms between runs on identical settings**, which is the honest reason
this section keeps telling you to measure rather than to copy these numbers. The path is not stable
minute to minute, and a default that fits at one median can miss at another.

Consequences, in order of how much they actually buy:

1. **`action_horizon: 16` is the default, and it is the measured answer rather than the predicted one.**
   The inference-only budget supports 8; the real path does not. 16 is also the ceiling — it is the
   chunk length the checkpoint was *trained* with, and the client refuses a horizon above it rather
   than silently executing a short chunk. The cost is reactivity: the policy re-plans half as often, so
   it corrects more slowly and drifts further open-loop between chunks.
2. **Shrink the frames if the tail is hurting you**, not to fix the median. It is also a
   *policy-quality* decision, not just plumbing: the model resizes internally anyway (640×480 costs
   only 5.5 ms more inference than 224×224), but pre-shrinking discards detail it might have used, and
   your dataset was recorded at one particular resolution. Score the result, do not assume.
3. **Move the policy closer** — `inference_site=edge`, which needs a GPU on the edge machine and the
   platform this repo does not create. On this WAN path **a 30 Hz closed loop is not reachable at any
   horizon the checkpoint supports**; ~12 Hz is what it does. If you need 30 Hz, you need the policy
   on the same machine.

`denoising_steps` is the biggest lever on *inference* specifically, and it must match what stage 03
evaluated at, or you are driving hardware with a model configuration you never scored.

**Set the receive timeout from your own measurement.** It must sit above your round-trip p99 or
episodes abort on healthy jitter — at `policy_timeout_ms: 1000` one request in fifty timed out and
was retried, which is why the default is 1500. Its real purpose is the arithmetic: the arm can hold a
stale pose for at most `(policy_retries + 1) × policy_timeout_ms`, so the defaults bound it at 3 s.
Upstream's client defaults to 15 s, which at this loop's round trip is not a timeout but an outage. On
a timeout the arm **holds**, the request is retried, and when the retries run out the episode aborts
and the arm is returned and released — hold, retry, then release. Holding briefly is right; going limp
on one dropped packet is worse, because the arm falls under gravity and drops whatever it is holding.

Run a `no_motion` rehearsal, read `chunk_latency` out of `results.json`, and set `action_horizon` and
`policy_timeout_ms` from *those* numbers before a powered episode.

### Cloud or edge inference

`inference_site` switches where the policy runs. **`cloud` is the default and the only validated
path.**

```bash
--set inference_site=edge edge_policy_platform=<your-gpu-edge-platform>
```

**`edge` names a platform this repo does not create.** A GPU on a hybrid node needs a device-plugin
design — driver container, container toolchain, device plugin, scoped to whatever distribution the
on-premise machine runs — that does not exist here, and the reference machine has no GPU to validate
it against. The switch is built and the robot task is unchanged either way; adding the platform and
proving the branch is an operator's job. Do not read the presence of the flag as support for it.

### Your machine's device paths are not ours

`robot_port`, `front_camera` and `wrist_camera` are per-machine facts, so survey your own:

- **Two SO-101 arms are indistinguishable by USB serial** — same adapter, same six servos. *Supply
  voltage* tells them apart: the follower runs from a 12 V supply, the leader is bus-powered at ~5 V,
  and reading the voltage register needs no motion.
- **Two identical USB cameras produce an identical `/dev/v4l/by-id` string**, so their symlinks
  collide and only one of them appears there at all — which looks exactly like a missing camera. Use
  `/dev/v4l/by-path/`, which is USB-port-derived.
- **Each camera claims two `/dev/video*` nodes** and only one carries frames; the other is
  `Metadata Capture`. Confirm with `v4l2-ctl --all` rather than guessing from the numbering.
- **Front versus wrist can only be settled by looking at a frame.** With two identical cameras there
  is no other honest method.
- The platform's `allowed_mounts` must list every device the task names, and the task needs
  `privileged: true` — without it a mounted character device is present, correctly owned, and
  unopenable.

### Calibration is the correctness question, not a detail

The policy learned the **calibrated** joint representation of your recording: five arm joints in
degrees about the midpoint of each joint's calibrated range, and the gripper 0-100 across its own.
That mapping is defined by the LeRobot version that recorded the data, which is why `lerobot_commit`
is pinned — it is a correctness constraint, not hygiene.

The client never reimplements the mapping; it drives through LeRobot's own `SO101Follower`. Set
`calibration_localpath` to the calibration JSON your dataset was recorded with and the client will
assert the servos' EEPROM matches it before enabling torque. LeRobot writes calibration *into* the
motors, so that assertion is a direct check that the arm in front of you is in the frame the policy was
trained in. Leave it empty and the calibration is read from the servos instead, which is the same data
without the cross-check.

If the arm has been recalibrated since recording, **a fresh calibration is a different joint space** and
the policy will drive in it wrongly. Recalibrating does not fix this; finding the original calibration
does.

### Jitter: this policy does not hold still, and the error metrics cannot see it

**"Reversals" means: the fraction of commands at which a joint changes direction.** For
consecutive commanded positions `v[i-1], v[i], v[i+1]`, sample `i` is a reversal when
`(v[i] - v[i-1]) * (v[i+1] - v[i]) < 0` — the joint was travelling one way and now travels
the other.

```
10.0 -> 10.2 -> 10.4 -> 10.6      0%     monotonic, going somewhere
10.0 -> 10.3 -> 10.1 -> 10.4    ~100%    changing direction every step
```

The metric is deliberately **scale-free**: a slow smooth move and a fast smooth move both
score near zero, and only oscillation scores high. Amplitude alone cannot separate
"traversing the workspace" from "vibrating in place", and vibrating in place is what
overheats a servo.

What the scale means, from synthetic references:

| signal | reversals |
|---|---|
| clean monotonic ramp | 0% |
| **teleop recordings (this dataset)** | **0-0.6%** |
| random walk | 50% |
| noise superimposed on a slow ramp | 62% |
| uncorrelated noise | 67% (analytically 2/3) |

So a policy in the 40-70% band is emitting per-step increments with almost no trend — close
to uncorrelated. That is the diagnosis, not just a large number.

Measured on the fine-tuned checkpoint, against the recordings it was trained on:

| | direction reversals, worst joint | per-step size, p50 |
|---|---|---|
| Teleop recordings (ground truth) | **0.6%** | 0.00-0.18 |
| Policy output | **43-68%** | 0.15-0.58 |

The recordings are smooth and monotonic, and hold four of the six joints *exactly* still
more than half the time. The policy reverses direction on roughly half of all consecutive
steps instead. The amplitude is small — a few tenths of a degree — so **the arm does the
same amount of servo work while oscillating rather than progressing.**

That is not cosmetic. The first long hardware run aborted after two minutes because the
**gripper reached 60 °C** while every other joint sat at its resting temperature: it is the
smallest servo with the least thermal mass, and it was being driven continuously through
dwell periods it should have spent stationary. At 70 °C these servos cut their own torque
and drop the arm — the one failure mode software cannot intervene in.

Two things this rules out, both measured rather than assumed:

- **Raising `denoising_steps` does not help.** Swept 4 → 8 → 16 → 32: reversals stayed at
  24-68% throughout while latency grew from 143 ms to 725 ms. Do not spend the budget there.
  (8 *is* the accuracy optimum — MSE 6.97 → 4.01 — for +83 ms, which is worth having on its
  own merits.)
- **It is not the re-planning.** Reversals within a single predicted chunk (57%) match those
  at chunk boundaries (55%), so the chunk itself is jittery, not the seam between chunks.

The mitigation is `command_deadband_deg` / `command_deadband_gripper`: hold the previous
command until the target moves more than the threshold, which reproduces the dwell the
recordings have. Replayed against a real run's commands:

| deadband (arm / gripper) | reversals | arm travel kept | gripper travel kept |
|---|---|---|---|
| off | 39-60% | 100% | 100% |
| 0.3 / 0.5 | 11-21% | 79-96% | 75% |
| **0.5 / 1.0** (default) | **2-12%** | **74-91%** | 48% |
| 1.0 / 2.0 | 0-3% | 34-86% | 34% |

A deadband costs nothing for sustained motion — it accumulates past the threshold within a
step or two, so this is a quantization, not a lag, which is why it is preferred to a
low-pass filter in a loop that already spends 400 ms per round trip. `held_by_deadband` in
`results.json` records how often each joint was held.

**Stage 03 now measures this** and warns above `smoothness_warn_reversals`. It does not fail
the stage, because a jittery policy is still drivable with the deadband — but it is no longer
something you discover from a hot servo.

### Thermal duty cycle: the run rests, it does not stop

These servos cut their own torque at 70 °C and drop the arm, which is the one failure mode
no software can intervene in. So the client reads `Present_Temperature` once per re-plan and,
at `max_servo_temp_c`, **rests rather than ending the run**: the arm returns to where the
episode began, torque comes off, and it waits until every servo is back under
`resume_servo_temp_c` before re-attempting the episode.

Three details that matter:

- **Torque must come off to cool.** A servo holding a pose against gravity keeps drawing
  current, so a rest that merely stopped sending actions would not cool anything. The arm is
  limp for the duration.
- **A rest re-attempts the episode rather than consuming it**, so `max_episodes` counts real
  attempts and a hot arm costs wall clock, not coverage. Budget `exec_timeout` accordingly —
  roughly a minute per rest from 55 °C to 45 °C with torque off.
- **`resume_servo_temp_c` is well below `max_servo_temp_c` on purpose.** Resuming the moment
  the ceiling cleared would stop and start every few seconds; the gap is what makes a rest a
  rest.

`max_cooldowns` bounds it, so an arm that cannot sustain the duty cycle stops instead of
cycling forever. Measured with the deadband on, three 60 s episodes peaked at 53 °C and
recovered between episodes rather than accumulating — against 60 °C and climbing in two
episodes without it.

**Do not raise `max_servo_temp_c` to make a run finish.** The headroom to 70 °C is what turns
a thermal stop into a controlled return instead of the arm going limp mid-motion.

### Reading `results.json`

Beyond the per-episode records:

| Field | Read it for |
|---|---|
| `units_check` | Per joint: raw ticks, LeRobot's value, ours, and the difference. All zeros is what you want |
| `envelope` | The clamp actually applied, per joint, in policy units |
| `resting_pose_outside_envelope` | The arm was resting past where it was calibrated to sweep. The first action gets clamped |
| `warmup_actions_outside_envelope` | The policy is asking for poses the arm cannot reach — investigate before a powered run |
| `chunk_latency` | End-to-end p50/p95/p99 of the real path. Compare against the budget table above |
| `policy_timeouts` | Non-zero means the network tail is biting. Raise `action_horizon` or move the policy closer |
| `loop_overruns`, `worst_loop_overrun_ms` | The loop could not hold `control_hz` |
| `clipped_by_joint` | How often each joint's command hit the envelope |
| `mean_shortfall` | Per joint, how far short of its command the arm actually stopped |
| `held_by_deadband` | Per joint, how many commands the deadband suppressed as dither |
| `max_temp_c`, `hottest_temps` | Hottest reading seen, and the per-joint readings at that moment — the maximum alone does not say where the heat is, and on this arm it has been one joint at a time |
| `rests`, `rested_s` | How often the run stopped to cool, and for how long in total |
| `attempt` (per episode) | Higher than `episode` where a thermal rest forced a re-attempt |

That last one matters more than it looks. **This arm does not reach what it is told**, and unevenly:
one joint achieved 86% of its commanded amplitude and another only 48%, and the elbow descends more
readily than it lifts. So an open-loop replay of absolute targets undershoots per joint, and slower
motion tracks better — which means `control_hz` and tracking error trade against each other.

## Sizing, and why it bites

`storage` is a **scheduled** resource: under-request it and kubelet evicts the pod mid-run;
over-request it and the submit is rejected for exceeding the node. So the size of your dataset is an
input, not something the workflow can discover — `dataset_size_gi` on every stage.

Stage 02's default `storage` budgets for the toolchain (~15 GiB), the base model (~15 GiB), the final
export (~12 GiB) and six checkpoints. **A checkpoint is ~24 GiB** — roughly twice the model — because it
carries fp32 optimizer state for every trainable parameter. Six, not the five that are retained: the
launcher deletes the oldest checkpoint only after writing the newest, so peak usage is one above the
limit, and that transient is what evicts a long run that budgeted for five.

## Measured on one 48 GB GPU

| | |
|---|---|
| Trainable parameters | 1,620,515,968 (51.5% of the model) |
| Step time, sustained | ~2.0 s/step at `global_batch_size: 32` |
| 10,000 steps, end to end | 5 h 52 m, including ten checkpoint saves and a 140 GB upload |
| Training loss | 1.141 → 0.018 over 10,000 steps (one epoch of 43 episodes) |
| Checkpoint size | ~24 GiB |

It fits one card because the visual tower and the language backbone stay frozen — the upstream default,
and what upstream's own SO-101 recipe uses. It fits without much to spare: raising `global_batch_size`,
or setting `tune_visual` or `tune_llm`, needs a multi-GPU node, where the trainer shards optimizer state
automatically.

To watch GPU memory while a run is in flight, query the cluster's DCGM exporter rather than instrumenting
the workflow — `DCGM_FI_DEV_FB_USED` is labelled by pod and container, so it attributes memory to the task
that is using it:

```bash
kubectl -n gpu-operator port-forward svc/nvidia-dcgm-exporter 9400:9400
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_FB_USED
```

Note this is a live gauge with no retention here, so it answers "how much headroom right now", not "what
did last week's run peak at". A true per-process peak would need GPU accounting mode enabled on the node,
which a workflow cannot set. The launcher logs `Trainable parameters` at startup; check it against what you intended
before letting a long run proceed.

For cost, multiply your GPU node's hourly price by the wall clock in the table at the top. Nothing here
scales a node group up or down for you.

## Validation status

| Stage | Status | Notes |
|---|---|---|
| 01 prepare | Validated | Both acquisition paths, v2 and v3 inputs, `model=none` and `gr00t-n17` |
| 02 fine-tune | Validated at 10,000 steps | Loss 1.141 → 0.018 in 5 h 52 m; all five retained checkpoints verified against their manifests |
| 03 open-loop eval | Validated | Against 10-step, 6,000-step and 10,000-step checkpoints — the error metric separates them by two orders of magnitude. **It does not measure smoothness**, which is why that is now reported separately |
| 04 HIL, `no_motion` rehearsal | Validated on real hardware | Both tasks gang-scheduled, the client resolved the policy task's host and connected, EEPROM matched the recording calibration, unit conversion agreed with LeRobot on all six joints, both cameras opened at the dataset's resolution, and the loop ran 29 replans over 30 s with zero writes to the servo bus |
| 04 HIL, powered episode | Validated on real hardware | One 15 s episode at capped speed: torque engaged holding the present pose, the policy drove the arm closed-loop for 254 steps over 16 replans at 16.9 Hz, the arm returned to within 1 deg of its start pose on every joint, and torque was released and verified off on all six |
| 04 HIL, `inference_site=edge` | **Not validated, and cannot be here** | Needs a GPU on the edge machine and a platform this repo does not create |
| 04 HIL, 30 Hz closed loop | **Not reachable over this WAN** | 15.2 Hz at the default horizon. Not payload-bound and not inference-bound — see the latency section |
| 04 HIL, sustained multi-episode run | **Not yet achieved** | A 20-episode attempt aborted 2 minutes in on gripper temperature, before the deadband existed. The abort worked as designed and the servo cooled with no damage; the run has not been repeated since |
| Task success | **Unmeasured** | No episode has been scored. There is no sensor for it — success is the operator's verdict, recorded through the control directory |

A reader with no robot gets the complete pipeline through stage 03. That is deliberate.

## Extending this

Stage 01's model-specific work is one conditional block, selected by `model`, with per-model scripts
under `files/`; adding a model touches that block rather than the whole file. Stages 02 and 03 wrap
GR00T's own launcher and evaluator, so a different model means different workflows, not different flags.

The GR00T toolchain is installed at run time by `files/gr00t/install_n17.sh` rather than baked into an
image, so the commit it pins is changed in one place. That file is injected by every stage.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Input download fails with `NoSuchBucket` on `REPLACE-WITH-…` | Overrides split across repeated `--set` flags, so only the last applied | One `--set`, all pairs. Confirm with `--dry-run` |
| `uv sync` fails: `Invalid zip file structure` | A wheel tracked in Git LFS was left as a pointer file | `git-lfs` must be installed in the task image; `install_n17.sh` fetches the content |
| HTTP 403 downloading the model | The Hub account has not accepted the gated backbone repo's terms | Accept them; the token's scope is not the problem |
| Bus error / "insufficient shared memory" at step 0 | `dataloader_num_workers > 0` with the default 64 MB `/dev/shm` | The platform must provide a `shared_memory` pod template — see `002-setup/values/` |
| Pod evicted part-way through a run | `storage` under-requested | Raise `dataset_size_gi` / `train_storage_gi` |
| Task COMPLETED but files missing in S3 | Output written unreadable to the uploading sidecar | Stage 02 ships a `manifest.json` inside each checkpoint and stage 03 verifies the download against it — that is where truncation surfaces |
| `osmo workflow status` is not a command | It is `osmo workflow query` | — |
| Stage 04: the client sits at "still waiting for the policy server" and eventually aborts | The server task is still downloading the checkpoint, still loading the model — or it failed. The group is gang-scheduled, so both start together | Read the **policy-server** task's own logs. `policy_wait_s` in `results.json` records how long it took |
| Stage 04 server dies with `Could not load this library: libtorchcodec_core*.so` | `install_n17.sh` verifies its toolchain by importing torchcodec, which needs ffmpeg — even though serving a policy decodes no video | `apt-get install ffmpeg` in the task, as every other stage does |
| A servo gets hot fast while the others stay cool, or stage 04 keeps resting to cool | The policy's dither, concentrated in the smallest servo — see **Jitter** | Set `command_deadband_deg` / `command_deadband_gripper`. Do not raise `max_servo_temp_c`; at 70 °C the servo cuts its own torque and drops the arm |
| Stage 04 stops with "reached the N-rest limit" | The arm cannot sustain this duty cycle | Shorten `episode_timeout_s`, widen the deadband, or lower `speed`. Raising `max_cooldowns` only makes it cycle longer |
| Stage 04 rests repeatedly with implausible temperatures, and `rested_s` is near zero | Corrupted serial reads, not hot servos. A rest that ends instantly is the tell — the re-read came back normal | Already guarded: readings above 90 °C are discarded and a rest needs `temp_confirm_reads` consecutive confirmations. `bad_temp_reads` counts the rejects |
| Stage 04 stops with "servos did not cool within" | Something is still drawing current with torque off | Check the arm is not resting against a hard stop or holding an object under load |
| Stage 03 warns about direction reversals | The checkpoint's chunks are jittery. MSE and MAE cannot see this | Drive stage 04 with the deadband set. Raising `denoising_steps` does not help — measured flat from 4 to 32 |
| Stage 04: the arm pauses at every chunk boundary, `loop_overruns` ≈ `replans` | A round trip costs more than one chunk of motion | The latency section above. Shrink the frames first, then raise `action_horizon` |
| Stage 04 fails with `Server error: <modality> key '...' must be in observation` | The server rejected the observation. `observation_shape` in `results.json` records the exact wire shape the server receives, with each key's *type* — a key arriving as `bytes` rather than `str` reads on the server as a key that is simply absent | Compare `observation_shape` against the server's own `action/state/video keys` log lines. Seen once and not reproduced by an identical resubmission |
| A stage 04 server-side error appears in the client log as one line with no server traceback | The server prints a traceback and keeps serving, so its stdout is a pipe that stays open and block-buffers | Fixed: the server runs under `python -u`. Anything older needs `PYTHONUNBUFFERED=1` |
| Stage 04 aborts with "did not answer within N ms" on a healthy network | `policy_timeout_ms` is below your round-trip p99 | Read `chunk_latency` from a `no_motion` run and set the timeout above its p99 |
| Stage 04 refuses to start: "the servos' EEPROM does not match the calibration file" | The arm has been recalibrated since the dataset was recorded, so the policy's joint frame is not this arm's | Find the original calibration. Recalibrating does not fix it — a fresh calibration is a *different* joint space |
| Stage 04 warns that the resting pose is outside the safety envelope | The arm is resting past where it was calibrated to sweep | Harmless — the first actions are clamped — but move the arm into the workspace the episodes covered, or the policy is extrapolating from a pose it never saw |
| Stage 04 fails with `Failed to write 'Lock' ... There is no status packet!` | A transient servo-bus failure. It has been seen inside LeRobot's `configure()`, whose `torque_disabled()` context re-enables torque from a `finally` — so a failure there can leave torque on | The client retries the engage and forces torque off between attempts. If it persists, check the arm's supply: the symptom is consistent with a servo browning out and rebooting |
| Stage 04 ends with the arm limp but not where it started | The return ramp finished sending goals before the arm arrived | Raise `release_settle_s`. `return_residual` in `results.json` is how far short it stopped |
| `results.json` says `torque_released: false` | Software could not confirm the servos are de-energised | **Cut power to the arm.** Then read `final_torque_state` to see which joints |
| A device is present inside the task, correctly owned, and cannot be opened | The container device cgroup | `privileged: true` on the task; the platform's `privileged_allowed` only grants the possibility |
| `osmo workflow validate` fails on Jinja where `--dry-run` passed | `validate` runs the full server-side render; `--dry-run` is a weaker single pass | Trust `validate`. In particular, a literal Jinja tag inside a comment survives `--dry-run` and fails `validate` |

Data-access validation is not an existence check: OSMO reports success for a bucket that does not
exist, and the failure surfaces at download instead.
