# Measured on this hardware

Figures from real runs on one 48 GB L40S and one SO-101 arm, carried over from the two per-model
pipelines this one replaced. They are here because they cost GPU hours and bench time to obtain and
cannot be re-derived from the code — not because they are current. **Treat them as a starting point
for sizing, and re-measure on your own hardware.**

The two "task performance" rows are the important ones to read before anyone concludes the pipeline
works: both policies drive the loop safely and score well against recorded trajectories, and neither
performs the task usefully.

---

## GR00T N1.7

One 48 GB card, `global_batch_size: 32`, visual tower and language backbone frozen (upstream's
default, and what upstream's own SO-101 recipe uses):

| | |
|---|---|
| Trainable parameters | 1,620,515,968 (51.5% of the model) |
| Step time, sustained | ~2.0 s/step |
| 10,000 steps, end to end | 5 h 52 m, including ten checkpoint saves and a 140 GB upload |
| Training loss | 1.141 → 0.018 over 10,000 steps (one epoch of 43 episodes) |
| Checkpoint size | ~24 GiB |

It fits one card only because those two components stay frozen, and it fits without much to spare:
raising the batch size, or unfreezing either, needs a multi-GPU node where the trainer shards
optimizer state automatically.

## pi0-FAST

One L40S, **44.39 GiB usable** (not the 46 GB `nvidia-smi` advertises), two 480×640 cameras, 6 action
dims, `bfloat16`, no gradient checkpointing:

| batch | peak VRAM | step time | samples/s |
|---|---|---|---|
| 1 | 36.02 GB | 0.465 s | 2.15 |
| 2 | 36.05 GB | 0.558 s | 3.58 |
| **4** | **39.71 GB** | 0.746 s | 5.36 |
| 6 | — | — | `torch.OutOfMemoryError` |

**There is no LoRA path** — LeRobot's `pi0_fast` is full fine-tune only, so a 7.2 GB model carries
~15 GB of optimizer state, and that fixed ~22 GB is what caps the batch. Each extra sample costs
~1.83 GB. Batch 6 fails asking for 182 MiB with 9 MiB free and 2.73 GiB lost to allocator
fragmentation, so it is at the edge rather than far past it. Two untried levers:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and gradient checkpointing.

`dtype` must be set to `bfloat16` explicitly; upstream defaults to `float32`, which roughly doubles
memory on a 2.6B model.

### Steps do not compare across models — budget in epochs

At batch 4, one epoch of a 43-episode, 18,744-frame dataset is **4,686 steps**:

| | steps | × batch | samples | epochs | wall clock |
|---|---|---|---|---|---|
| GR00T, comparable run | 10,000 | 32 | 320,000 | 17.1 | ~6 h |
| pi0-FAST at 10,000 | 10,000 | 4 | 40,000 | 2.1 | ~2.1 h |
| pi0-FAST at 30,000 | 30,000 | 4 | 120,000 | 6.4 | ~6.2 h |
| pi0-FAST at 80,000 | 80,000 | 4 | 320,000 | 17.1 | ~16.6 h |

Two traps that follow from this. A cosine schedule auto-scales *down* when the step count is below
its decay length but does **not** scale up — left at a 30,000 default while training to 80,000, the
learning rate hits its floor at step 30,000 and stays there for the remaining 50,000. And a short
run is dominated by start-up and reports roughly twice the throughput a sustained one achieves, so
never budget `exec_timeout` from one: a task killed by its timeout publishes nothing.

## pi0-FAST inference latency

Decoding is autoregressive, so latency depends on what the policy emits, and **how long the
checkpoint trained moves the figure by an order of magnitude in both directions**:

| checkpoint | local p50 | note |
|---|---|---|
| barely trained (200 steps) | ~4.6 s | never emits the stop token; decodes to its cap |
| under-trained (constant pose) | 161 ms | stops immediately — measures fast, means nothing |
| trained, 40,000 steps | ~407 ms | from a 664 ms round trip less ~257 ms of network |
| trained, 17 epochs | ~546 ms | from an 804 ms round trip less ~257 ms of network |

A multi-second round trip reads as a network or payload problem and is not: the same transport and
the same 1.84 MB payload measured 496 ms with another policy on the same link minutes apart.
**Before investigating latency, check how long the checkpoint trained.**

End to end, cloud GPU to on-premise arm, at `action_horizon: 16`:

| | under-trained | trained |
|---|---|---|
| Round trip p50 | 415–519 ms | 804 ms |
| Round trip p99 | 819–968 ms | 1,059 ms |
| Motion bought per chunk | 533 ms | 533 ms |
| Achieved control rate | 15.8 Hz | 11.6–12.8 Hz |

A chunk of N actions at `control_hz` buys `N / control_hz` of motion, and the arm pauses at a chunk
boundary if the round trip took longer — with a trained policy at horizon 16 it pauses at **every**
boundary, moving about 39% of the wall clock. Clearing the p99 needs `action_horizon >= 32`, at the
cost of running open-loop for 1.07 s between re-plans. The round trip is also mostly network
(~257 ms of pod-to-pod overhead), so co-locating the policy with the robot would help more than a
faster model. Inference peak VRAM is **7.31 GB**, which makes co-location plausible on a modest edge
GPU.

---

## What was validated, and what was not

Per-stage, from the two predecessor pipelines. Dates omitted; these describe what was demonstrated,
not what is guaranteed.

| Claim | Status | Evidence |
|---|---|---|
| Dataset port preserves content | Validated | 43 episodes and 18,744 frames preserved, asserted by count rather than by version string |
| Training refuses a wrong-version input | Validated | fails in ~1 min naming the prepare stage, rather than dying inside a loader |
| GR00T trains and publishes | Validated at 10,000 steps | loss 1.141 → 0.018 in 5 h 52 m; all five retained checkpoints verified against their manifests |
| pi0-FAST trains and publishes | Validated at 80,000 steps | 16 h 55 m, 4 h 14 m per 20,000 steps, peak 39.71 GB, 0.749 s/step; final learning rate reached its floor, confirming the schedule completed |
| Offline scoring separates checkpoints | Validated | pi0-FAST at 80,000 steps: MSE 2.79, MAE 0.70. At 20,000: MSE 203.1 — training length dominates by 73× |
| Scoring guards fire | Validated | an out-of-range trajectory ID gives `complete: false` and a non-zero exit |
| HIL rehearsal | Validated on hardware | both tasks gang-scheduled, host token resolved, observation accepted, calibration matched the recording, both cameras opened at the dataset's resolution, zero writes to the servo bus |
| HIL powered, GR00T | Validated on hardware | 254 steps over 16 replans at 16.9 Hz, returned to within 1° on every joint, torque released and verified off on all six |
| HIL powered, pi0-FAST | Validated on hardware | three 60 s intervals, 150 s of driving, returned to 1.14°, peak 37 °C, no thermal rests, torque verified off |
| Policy co-located with the robot | **Not validated** | needs a GPU on the arm machine, which this hardware does not have |
| 30 Hz closed loop | **Not reachable over this link** | ~15 Hz at the default horizon; neither payload-bound nor inference-bound |
| Sustained multi-interval run | **Not achieved** | a 20-interval attempt aborted 2 min in on gripper temperature, before the command deadband existed. The abort worked as designed and the servo cooled undamaged; not repeated since |
| **Task performance** | **Poor, and not addressed** | both policies run the loop safely and score well against recorded trajectories; neither performs the task usefully on real hardware. Likely causes: 43 episodes is under-provisioned — SmolVLA's own docs call 25 insufficient and 50 a floor — and `action_horizon` 16 makes the arm pause at every chunk boundary |
| Task success, as a number | **Not measured, by design** | whether a policy does the task is judged at the bench by the person watching. Nothing here records a verdict, because a number would look like a measurement and there is no sensor behind it |

## Watching GPU memory during a run

Query the cluster's DCGM exporter rather than instrumenting the workflow — `DCGM_FI_DEV_FB_USED` is
labelled by pod and container, so it attributes memory to the task using it:

```bash
kubectl -n gpu-operator port-forward svc/nvidia-dcgm-exporter 9400:9400
curl -s localhost:9400/metrics | grep DCGM_FI_DEV_FB_USED
```

This is a live gauge with no retention, so it answers "how much headroom right now", not "what did
last week's run peak at". A true per-process peak needs GPU accounting mode enabled on the node,
which a workflow cannot set.
