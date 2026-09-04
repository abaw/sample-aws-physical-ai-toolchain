# SO-101 VLA fine-tuning on OSMO

Fine-tune a vision-language-action model on your own SO-101 robot recordings, and judge the result
before it ever drives hardware — three OSMO workflows, one GPU, entirely in the cloud.

The model is [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T). The dataset is whatever your SO-101
recorded, in [LeRobot](https://github.com/huggingface/lerobot) format.

| Stage | Workflow | GPU | Measured wall clock | Publishes |
|---|---|:---:|---|---|
| 01 | `01_prepare_dataset.yaml` | none needed | ~4 min | a validated dataset at a stable prefix |
| 02 | `02_finetune.yaml` | 1 | ~10 min for a 10-step smoke run; ~3 h for 10k steps | checkpoints, per run |
| 03 | `03_openloop_eval.yaml` | 1 | ~9 min | per-dimension plots and error metrics |

Run them in order. Each reads what the previous published, so the seam between them is an S3 prefix
rather than a shared filesystem — any stage can be re-run alone.

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

Then the real run, with `max_steps` in the 10k-20k range. Budget `exec_timeout` from the smoke run's
step rate: **a task killed by its timeout publishes nothing.**

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

Use episodes the fine-tune did not train on, and use several — one lucky trajectory is not convergence.

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

For calibration, the 10-step smoke checkpoint above scores average **MSE ≈ 1347, MAE ≈ 30** on this
example's dataset. That is the floor — a useful run must be far below it. If a 10-step checkpoint ever
looks *good*, suspect the evaluation, not the model.

`execution_horizon` and `denoising_steps` are exposed here because they are also the two knobs that set
a real-time control loop's latency budget. Evaluate at the values you intend to drive an arm with.

## Sizing, and why it bites

`storage` is a **scheduled** resource: under-request it and kubelet evicts the pod mid-run;
over-request it and the submit is rejected for exceeding the node. So the size of your dataset is an
input, not something the workflow can discover — `dataset_size_gi` on every stage.

Stage 02's default `storage` budgets for the toolchain (~15 GiB), the base model (~15 GiB), the final
export (~12 GiB) and five retained checkpoints. **A checkpoint is ~24 GiB** — roughly twice the model —
because it carries fp32 optimizer state for every trainable parameter.

## Measured on one 48 GB GPU

| | |
|---|---|
| Trainable parameters | 1,620,515,968 (51.5% of the model) |
| Step time, steady state | ~1.0 s/step |
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
| 02 fine-tune | Validated at 10 steps | A full 10k-step run has not been done |
| 03 open-loop eval | Validated | Against a 10-step checkpoint, i.e. as a negative control |
| Hardware-in-the-loop | Not included | This example ends at open-loop evaluation |

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

Data-access validation is not an existence check: OSMO reports success for a bucket that does not
exist, and the failure surfaces at download instead.
