# SO-101 + pi0-FAST on OSMO

Fine-tune [pi0-FAST](https://github.com/Physical-Intelligence/openpi) on your own SO-101 recordings,
through [LeRobot](https://github.com/huggingface/lerobot)'s `pi0_fast` policy, on one GPU.

**This is a second model beside `so101-vla/`, not a replacement for it.** That example runs
[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T), which denoises actions with a flow-matching
head. pi0-FAST emits **discretized action tokens autoregressively** — a different architecture,
a different training entry point (`lerobot-train` rather than GR00T's own scripts), and a different
dataset format. Having both is how you find out which parts of a pipeline were shaped around one
model without anyone noticing.

| Stage | Workflow | GPU | Measured wall clock | Publishes |
|---|---|:---:|---|---|
| 01 | `01_port_dataset.yaml` | none | ~2 min | a v3.0 dataset at a stable prefix |
| 02 | `02_finetune.yaml` | 1 | ~4.2 h at the default 20k steps | checkpoints, per run |

Stages 03 (open-loop evaluation) and 04 (hardware-in-the-loop) are **not written yet**. Until they
are, this example trains a policy and does not evaluate or deploy one.

## Why stage 01 exists

The two model stacks want opposite dataset formats and **neither can read the other's**:

| | GR00T N1.7 | LeRobot `pi0_fast` |
|---|---|---|
| Format | v2 + `meta/modality.json` | v3.0 |
| Older versions | converts v3 down to v2 | `BackwardCompatibilityError` |

So `so101-vla/01_prepare_dataset.yaml` acquires and validates a dataset and adapts it for GR00T;
this stage takes that published tree and changes only the format. It does not re-acquire or
re-validate. Point it at any prepared v2.1 tree.

**Use a `dataset_out_name` no other pipeline uses.** A dataset prefix is named only by that value —
no model, no format in the path — so reusing an existing name overwrites that tree. The workflow
refuses the placeholder for this reason, but it cannot know which names are already taken.

If your recording is already v3.0, stage 01 detects that and republishes it unchanged, so it is safe
to run unconditionally.

### The v2.1 datasets that are secretly v2.0

A dataset written by GR00T-flavoured tooling may declare `codebase_version: v2.1` while carrying the
**v2.0** metadata layout — an aggregate `meta/stats.json` and no per-episode
`meta/episodes_stats.jsonl`. LeRobot's v3 converter validates the version string, believes it, and
dies on the missing file. LeRobot used to ship a v2.0 → v2.1 script that wrote it; that script is
gone as of v0.6.x, so `files/pi0fast/backfill_episodes_stats.py` stands in for it.

The backfill computes per-episode statistics with LeRobot's own `get_feature_stats`, so shapes and
the quantile set match what the converter expects. **Video features get shape-correct placeholders
rather than measured values**, which is sound only because `PI0FastConfig` sets
`VISUAL: NormalizationMode.IDENTITY` — pi0-FAST does not normalise images and never reads them. Read
the script's docstring before reusing this for a policy that does.

## Prerequisites

**A pool with a GPU platform.** Stage 02 defaults to `platform: l40s`; override with
`--set platform=`. Stage 01 needs no GPU and does not request one.

**A Hugging Face token, as an OSMO credential:**

```bash
osmo credential set huggingface_token --type GENERIC --payload-file token=<file-holding-the-token>
```

pi0-FAST's text tokenizer is `google/paligemma-3b-pt-224`, a **gated** repo. The account behind the
token must have visited that page and accepted the terms — a token with the right scope still gets
HTTP 403 until someone clicks through. Stage 02 preflights this with `curl` before installing
anything, so the failure arrives in seconds rather than four minutes in.

**A prepared LeRobot dataset**, v2.1 or v3.0. `so101-vla/01_prepare_dataset.yaml` produces one.

## Running it

```bash
B=s3://<bucket>/so101-pi0fast

# 1. port the dataset to v3.0
osmo workflow submit 003-workflows/so101-pi0fast/01_port_dataset.yaml \
  --set storage_url=$B \
        dataset_url=s3://<bucket>/so101-vla/dataset/<name>/ \
        dataset_out_name=<a-new-name>

# 2. fine-tune. Smoke-test with steps=50 first: it exercises the whole path
#    — model download, dataset load, optimizer step, checkpoint upload — in minutes.
osmo workflow submit 003-workflows/so101-pi0fast/02_finetune.yaml \
  --set storage_url=$B dataset_url=$B/dataset/<a-new-name>/ \
        steps=50 save_freq=50 exec_timeout=1h
```

Every override goes in a **single** `--set`. Repeating the flag is not additive — only the last group
applies, and the run proceeds on the placeholder defaults.

Checkpoints land under `$B/checkpoints/pi0fast/<workflow-id>/`, one prefix per run, so two
experiments cannot overwrite each other. Each published directory carries a `manifest.json` of its
own files and sizes, because OSMO's uploader logs per-file failures and then reports success anyway,
and a task cannot check its own upload — the uploader runs after the container exits.

## Sizing, and why batch 4

Measured on one L40S (**44.39 GiB usable**, not the 46 GB `nvidia-smi` advertises), two 480×640
cameras, 6 action dims, `bfloat16`, no gradient checkpointing:

| batch | peak VRAM | step time | samples/s |
|---|---|---|---|
| 1 | 36.02 GB | 0.465 s | 2.15 |
| 2 | 36.05 GB | 0.558 s | 3.58 |
| **4** | **39.71 GB** | 0.746 s | 5.36 |
| 6 | — | — | `torch.OutOfMemoryError` |

**There is no LoRA path.** LeRobot's `pi0_fast` is full fine-tune only, so a 7.2 GB model carries
~15 GB of optimizer state, and that fixed ~22 GB is what caps the batch — each extra sample costs
about 1.83 GB. Batch 6 fails asking for 182 MiB with 9 MiB free, with 2.73 GiB lost to allocator
fragmentation, so it is at the edge rather than far past it. Two untried levers:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and `--policy.gradient_checkpointing=true`.

**`dtype` is set explicitly to `bfloat16`** because upstream defaults it to `float32`, which roughly
doubles memory on a 2.6B model. Inheriting that silently is how a run OOMs for no visible reason.

### Budget steps in epochs, not steps

At batch 4 one epoch of a 43-episode, 18,744-frame dataset is **4,686 steps**. The default 20,000
steps is ~4.3 epochs and ~4.2 h. A comparable GR00T run covered ~17 epochs, which here would be
~80,000 steps and ~16 h. Raise `steps` deliberately and raise `exec_timeout` with it.

**Do not budget `exec_timeout` from a short run's step rate.** A brief run is dominated by start-up
and reports roughly twice the throughput a sustained one achieves. **A task killed by its timeout
publishes nothing.**

## Inference latency, measured

Relevant to stages 03 and 04, and measured on one L40S over 50 timed calls:

| variant | p50 | actions/call | motion bought at 30 Hz | ratio |
|---|---|---|---|---|
| default | **160.7 ms** | 50 | 1,667 ms | **10.4×** |
| `use_kv_cache=false` | 6,708.7 ms | 50 | 1,667 ms | 0.25× |
| `torch.compile` | 161.7 ms | 50 | 1,667 ms | 10.3× |

Inference peak VRAM is **7.31 GB** — no optimizer state needed, so a serving task is far smaller
than a training one.

A chunk of N actions executed at `control_hz` buys `N / control_hz` of motion, and the arm only
stalls if inference took longer than that. At 10× headroom, **`n_action_steps` becomes a
control-quality choice rather than a way to hide latency** — executing all 50 means running
open-loop for 1.67 s, unable to react to a moved object.

**The KV cache is worth 42×.** It defaults on; never disable it to save memory, and at 7.31 GB there
would be no reason to. `torch.compile` showed no benefit here, but it compiled in 2.2 s, which is too
fast for a model this size and suggests the autoregressive decode loop was never traced — read that
as "no benefit from this approach", not "compilation cannot help".

## Validation status

Honest per stage, as of 2026-09-16:

| Claim | Status | Evidence |
|---|---|---|
| Stage 01 ports v2.1 → v3.0 | Validated | 43 episodes and 18,744 frames preserved; counts asserted by the workflow, not just the version string |
| Stage 01 leaves the source tree alone | Validated | the GR00T v2.1 tree and its `modality.json` unchanged afterwards |
| Stage 02 refuses a v2.1 input | Validated | fails in ~1 min naming stage 01, rather than dying inside LeRobot's loader |
| Stage 02 trains and publishes a checkpoint | Validated at 50 steps | peak 39.71 GB, 0.743 s/step — reproduces the sizing table above |
| Stage 02 at the default 20,000 steps | **Not yet run** | the 4.2 h figure is extrapolated from the measured step time, not observed |
| Policy quality | **Unmeasured** | no open-loop evaluation stage exists yet, and no checkpoint has driven hardware |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: meta/episodes_stats.jsonl` | a v2.1 tree with the v2.0 metadata layout | run stage 01, which backfills it |
| `ERROR: LeRobot reads v3.0 only, found v2.1` | stage 02 given an unported tree | run stage 01 first |
| `RuntimeError: Failed to load required tokenizers` | the gated PaliGemma repo, 403 three exception layers down | accept the terms as the token's account; stage 02's preflight catches this earlier |
| `ValueError: 'repo_id' argument missing` | `policy.push_to_hub` defaults to true | already handled — both stages pass `--policy.push_to_hub=false` |
| `torchcodec ... cannot be loaded. Falling back to 'pyav'` | no `ffmpeg` in the image, so `libtorchcodec_core*.so` cannot link | already handled — both stages install `ffmpeg`. Left unfixed it trains on a different decode path than evaluation uses |
| `FileExistsError: Output directory ... already exists` | `lerobot-train` refuses a pre-existing output dir unless resuming | already handled — the dir is removed, not created |
| `ModuleNotFoundError` naming `lerobot[pi]` | that extra does not exist at v0.6.1, though `require_package` names it | install `transformers-dep,scipy-dep,dataset,training` — see `files/pi0fast/install_pi0fast.sh` |
| `torch.OutOfMemoryError` at batch ≥ 6 | 22 GB of weights plus optimizer state on a 44.39 GiB card | keep batch 4, or try gradient checkpointing |

## Pinning

`files/pi0fast/install_pi0fast.sh` pins LeRobot to a tag, not `main`. Between v0.5.1 and v0.6.1 this
project removed GR00T N1.5 support outright, so tracking `main` can break a pipeline between two runs
of the same workflow. Override with `LEROBOT_TAG` if you need a different one, and re-check
`pi0_fast`'s config defaults against whatever you pick.
