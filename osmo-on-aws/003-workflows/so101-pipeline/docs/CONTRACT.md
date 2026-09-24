# The workflow-to-model contract

What a directory under `models/` must provide for the **workflow** to run a model, and what the
workflow guarantees in return. Derived from the two pipelines that already exist — GR00T N1.7 and
pi0-FAST — rather than from one of them.

## Scope: there are three contracts, and this document is one of them

| Contract | Between | Defined by | Where |
|---|---|---|---|
| **A** | workflow ↔ model | the workflow | **this document** |
| **B** | robot ↔ policy server | **the robot** | `SO101_POLICY_INTERFACE.md` |
| C | workflow ↔ user | the workflow | the spec's variables and README |

**B is not the workflow's business.** A robot's degrees of freedom, state layout and camera set
determine what an observation and an action *are*; that is a property of the hardware, not of any
orchestration around it. The workflow's entire involvement is passing each side the arguments it
needs — checkpoint, dataset, port and camera names to the server; host, port, devices and cameras to
the client — and then getting out of the way. It neither defines nor validates the shapes.

**So "model-agnostic" means something narrower than it sounds**, and worth stating plainly: the
workflow has the contracts it needs to plug in a model. It does **not** mean a model can be plugged in
irrespective of the robot. A policy server must implement an adapter to whatever interface the target
robot defines, and a model that has no such adapter cannot drive that robot no matter how well it
satisfies this document. Contract B is out of scope for the workflow and in scope for the project,
because the project ships an SO-101 example.

## The principle

**The pipeline passes what it owns. The model owns everything else.**

| The pipeline owns | The model owns |
|---|---|
| where the data is, where to write | its launcher and the spelling of its flags |
| how many steps, how big a batch, how often to save, the seed | environment variables its launcher reads |
| which device each of the dataset's cameras is, the serial port, the policy port | its embodiment tag, denoising steps, dtype |
| trajectory ids, execution horizon | its modality config, its working directory |
| whether the arm may move | how many GPUs its launcher wants |

This split is why the contract is a set of **filenames**, not a set of flags. The two real launchers
have nothing in common: one takes `NUM_GPUS=… MAX_STEPS=… bash examples/finetune.sh --base-model-path …`
after changing directory, the other takes `lerobot-train --policy.type=… --batch_size=…`. Any attempt
to find a shared flag vocabulary fails on the first model added.

## The twelve files

A model directory is `models/<name>/`. The pipeline selects it by templating the localpath, so
**`pipeline.yaml` never names a model**. Adding one is a new directory.

**Every one of the twelve must exist.** A missing file fails at `osmo workflow validate`, naming the
path; an empty one validates and costs nothing. So a model that needs no helper ships an empty file,
and the pipeline needs no conditional to know which models have which.

```
install.sh           SOURCED by every stage. $1 stage (prepare|train|eval|serve)
prepare.sh           $1 dataset-in   $2 dir-out
train.sh             $1 dataset  $2 dir-out  $3 steps  $4 batch  $5 save-freq  $6 seed
score.sh             $1 ckpt  $2 dataset  $3 dir-out  $4 traj-ids  $5 horizon  $6 steps
serve.sh             $1 ckpt  $2 dataset  $3 port  $4 camera-names
client_transport.sh  SOURCED by the robot task; exports TRANSPORT_MODULE
common.py            injected into every stage        may be empty
prepare_lib.py       injected into prepare            may be empty
train_lib.py         injected into train              may be empty
score_lib.py         injected into eval               may be empty
serve_lib.py         injected into hil                may be empty
transport.py         injected into both hil tasks     may be empty
```

A helper lands beside the script that uses it, so a plain `import common` works. `common.py` is the
place for anything two stages must agree on — one model uses it for a camera-name mapping that
training, scoring and serving all have to apply identically.

Every script receives **every argument it might need, whether it uses it or not**. Ignoring an
argument is free; being unable to see one is a contract change. All paths are absolute.

A shell function defined in `install.sh` must be **exported** to reach a stage script: the stage
scripts are executed rather than sourced, so they inherit exported variables and functions only.

### `install.sh` — sourced, not executed

```
$1  stage name: prepare | train | eval | serve
```

Sourced by every stage before anything else, so it may export variables the other scripts need
(`GROOT_DIR`, a venv on `PATH`). **It takes the stage** so a model can install less for a cheap stage:
converting parquet does not need a training toolchain, and GR00T's is a git clone plus `git lfs pull`
plus `uv sync`. A model that wants one install for everything can ignore `$1`.

Must be idempotent and must leave the toolchain usable by the stage named.

### `prepare.sh` — convert a dataset into this model's format

```
$1  input dataset root   (read-only; contains meta/info.json)
$2  output directory     (exists; publish the converted tree here)
```

**Does NOT take the camera set.** A modality config must name the cameras *the dataset has*, and the
dataset's own feature list is the authority for that. A workflow variable could disagree with the
recording, and nothing downstream would notice. Derive camera names from
`meta/info.json`'s `observation.images.*` keys, never from a flag.

Must fail non-zero if conversion loses episodes or frames — a version bump alone is not evidence, as
both real converters demonstrate.

### `train.sh` — fine-tune

```
$1  dataset root      $4  batch size
$2  output directory  $5  save frequency, in steps
$3  steps             $6  seed
```

Six arguments because those are the six things a *user* of any model wants to set. Everything else —
learning rate, dtype, gradient checkpointing, GPU count, embodiment tag, modality config — belongs to
the model and lives in its script or its `model.vars`.

Must write a checkpoint the model's own `score.sh` and `serve.sh` can load, and should write
`$2/metrics.json` (see below) when it can.

### `score.sh` — evaluate offline, and produce the gate's input

```
$1  checkpoint           $4  trajectory ids ("3 7 11")
$2  dataset root         $5  execution horizon
$3  output directory     $6  max steps per trajectory
```

**Must write `$3/metrics.json`.** The pipeline gates on it and does not read logs. A model whose
upstream evaluator only prints — GR00T's does — scrapes its own log inside this script. The pipeline
must never learn another project's log format.

### `serve.sh` — serve the policy on the wire contract

```
$1  checkpoint    $3  port
$2  dataset root  $4  camera names, comma-separated ("front,wrist")
```

Gets the **dataset** because a policy served with different normalisation statistics than it trained
on drives confidently to the wrong pose, and because the camera-key check needs the checkpoint's own
feature list. pi0-FAST's server requires it; GR00T's does not.

Gets camera **names only — no device paths.** The server runs on a GPU node that has no cameras
attached; a device path there would be meaningless, and passing one would imply the server has
something to do with devices. It needs the names solely to refuse a rig the checkpoint was not
trained on, before torque. The pipeline strips the device half of the mapping.

Runs in the foreground until killed. The robot client is the `lead` task, so its exit tears this down.

### `client_transport.sh` — sourced by the robot task

Provisions whatever the shared robot client needs to reach this model's server, and exports
`TRANSPORT_MODULE` naming the importable module. Sourced, not executed, because it must affect the
caller's environment.

The transport is the **model's** choice; see `SO101_POLICY_INTERFACE.md`. A model whose transport is an
installable package installs it here and ships an empty `transport.py`. A model whose transport is its
own code puts it in `transport.py` and exports `TRANSPORT_MODULE=transport` — the module is named for
the slot, so the pipeline never learns a per-model name.

### `transport.py` — injected into both HIL tasks

One file, given to the policy server and the robot client alike, so the two cannot disagree about the
wire. May be empty.

Note it is a file, and a file cannot declare dependencies. If serving needs packages the base image
lacks, install them from `install.sh` gated on `$1` being `serve`.

## `metrics.json`

Written by `score.sh`, read by the pipeline's gate.

| Key | Required | The gate acts on it |
|---|---|---|
| `complete` | yes | **yes** — `false` fails the gate. Means every requested trajectory was actually scored; an out-of-range id must not quietly shrink the sample |
| `mse`, `mae` | yes | no — reported |
| `worst_reversals_predicted` | recommended | no — reported. Compare against the ground-truth figure |
| `worst_reversals_ground_truth` | recommended | no — it is the check on the metric: teleop data scores under 1%, so if it does not, the metric or the dataset is wrong rather than the policy |
| anything else | no | no |

**Only `complete` fails the gate, and deliberately only that.** It is the one key that is boolean, stable
and about the *measurement* rather than the policy: an out-of-range trajectory id must not quietly shrink
the sample an average is taken over.

Everything else is reported. Two reasons, both learned by measuring:

- **The error figures are not reproducible.** Three runs of one checkpoint over the same trajectories gave
  MSE 15.53, 13.70 and 11.49 — unseeded sampling — so a numeric threshold is a coin toss near the
  boundary, and comparing two models from single runs measures noise.
- **Whether a policy performs the task is not a question metrics answer.** That is settled by watching the
  arm. Offline scoring is for the failures that are cheap to catch offline — a wrong joint order, missing
  normalisation statistics, jitter that would cook a servo — not for judging performance.

## `model.vars` — optional

Shell assignments carrying values that belong to the model rather than to a run, which the *spec*
nonetheless has to know:

```sh
runtime_image="nvcr.io/nvidia/pytorch:25.04-py3"
hf_token_credential="huggingface_token"
```

**`runtime_image` is the only required one.** The pipeline's default is an unparseable placeholder,
because no image is right for every model and a default that submits and then fails inside the
container is worse than one rejected up front.

`hf_token_credential` names an OSMO credential exposed to the model's scripts as `HF_TOKEN`, and the
pipeline's empty default already works — a model overrides it only to pull from a gated repository. So
a model needing nothing else can be a one-line file.

`./submit.sh --model <name>` loads this file. Assembling `--set` by hand means passing these yourself,
and omitting `runtime_image` is only caught at submit, not at `--dry-run`.

## The robot interface is elsewhere

`serve.sh`'s **arguments** are contract A: the workflow hands it a checkpoint, a dataset, a port and
the camera names. What it *does* with them — speak the protocol the robot's client expects — is
contract B, and for the SO-101 example that is `SO101_POLICY_INTERFACE.md`.

The practical consequence for a model author: satisfying this document gets a model through prepare,
train and score. Driving an arm additionally requires an adapter for that robot's interface, and the
workflow cannot tell you whether you have one — it will happily start a server that speaks nothing the
client understands, and the symptom will be a client timing out.

## Output prefixes

```
<storage_url>/dataset/<model>/<dataset_name>/        converted dataset — keyed by NAME, so it is a cache
<storage_url>/checkpoints/<model>/<workflow_id>/     one prefix per run, so experiments cannot collide
<storage_url>/eval/<model>/<workflow_id>/
<storage_url>/hil/<model>/<workflow_id>/
```

The dataset prefix is keyed by **name, not run id**, deliberately: conversion is expensive, its result
is reusable, and a run-id-keyed prefix means re-converting every time and then hand-copying a
workflow-id-stamped path into the next submission. Conversion must therefore be idempotent.

Keying by `<model>` is what makes one model's preparation unable to destroy another's tree — the two
real stacks want mutually unreadable formats, and a prefix named only by dataset would have one
overwrite the other.

## Every payload is a Jinja template, whether you wanted that or not

OSMO renders Jinja over the **body** of every file injected via `files: localpath:`, not only over the
spec. That is a fact about the platform, not a rule of this contract, and it has two faces.

### Templating a payload is allowed — and usually a bad trade

It works. Verified in a payload: `{{ var }}`, filters, `{% if %}`, `{% for %}`, and OSMO's own
`{{output}}` and `{{workflow_id}}`.

**But the pass that renders payloads sees only `--set` values. `default-values:` are invisible to it.**
So a payload referencing a workflow variable makes that variable **mandatory `--set` input on every
submission, forever** — forget it and the submit fails with `UndefinedError`, and nothing in the spec
tells a user the requirement exists. That is a bad trade for almost any purpose, and it is why the
default advice is: **pass values as script arguments from the spec, not as template variables.**

The exception, and the one case worth using:

| Referencing | Cost | Verdict |
|---|---|---|
| `{{output}}`, `{{workflow_id}}` and OSMO's other runtime tokens | none — always in scope, cannot be forgotten | **fine** |
| a workflow variable | becomes mandatory `--set` on every submission | **discouraged** — take an argument instead |

`models/dummy-b/prepare.sh` shows the good case: it stamps `{{workflow_id}}` into a provenance file,
which needs no `--set` and records which run produced a tree.

Two further rules if you do template:

- **Avoid quote literals of either kind inside a Jinja expression.** The payload is embedded in a
  quoted context, so a single quote can terminate it and a double quote can arrive
  backslash-escaped — `{{ cam.split("=")[0] }}` fails with `unexpected char '\'`. Jinja's bare
  `.split()` splits on whitespace and needs no literal, so pass a space-separated string.
- **A bare statement tag in a comment is still parsed.** Writing `{% if %}` in prose is a syntax
  error that fails the whole spec. This trap has caught three separate edits here, including a comment
  whose subject was the trap.

A templated payload also cannot be run or tested outside OSMO, and a syntax error in one surfaces as a
400 with a line number pointing into an assembled template nobody can see.

### The accidental case is the dangerous one

Idiomatic bash contains Jinja openers by coincidence. `${#arr[@]}` is lexed as `{#`, an unterminated
comment. Alone that is a 400 at validate. Combined with a later `#}` anywhere in the file, it
**validates clean and silently deletes everything between the two** — no error at submit, validate, or
runtime. In a robot client that is how a `trap release_torque EXIT` disappears.

Use `${arr[*]}` with a counter, or wrap the region in a raw block. Run `lint_payloads.py`, which resolves
templated localpaths by globbing and flags openers in payloads that have not declared themselves. The
marker is a directive to that tool only — OSMO templates every payload regardless — and `yes` is its
only accepted value, so a typo fails loudly rather than quietly meaning "lint me".

## What a model must NOT do

- **No writing outside `$2`/`$3`.** The pipeline publishes those directories and nothing else.
- **No assuming the output directory is empty** for `train.sh` — one real launcher refuses a
  pre-existing output directory, so the pipeline removes rather than creates it.
- **No moving the arm.** Only the robot client does that, and only when told to explicitly.
