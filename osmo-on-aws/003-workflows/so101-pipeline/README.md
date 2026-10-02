# One pipeline, any model, one SO-101 arm

Fine-tune a vision-language-action model on your own SO-101 recordings, score it before it touches
hardware, and close the loop on the real arm — **one OSMO spec, four switchable stages**.

```bash
./submit.sh --example full-chain --model gr00t-n17 \
    storage_url=s3://<your-bucket>/so101 \
    source_dataset_url=s3://<your-bucket>/raw/<name>/ \
    dataset_name=<name> \
    lang_instruction="<the task text your dataset was annotated with>"
```

That runs prepare → train → eval → a HIL **rehearsal**. The arm does not move until you add
`allow_motion=true`.

`submit.sh` assembles the `--set` arguments, checks every key against the spec's own `default-values` so a
typo is refused rather than ignored, and prints the exact `osmo workflow submit` it runs. `--dry-run`
passes through. The `examples/*.vars` files are plain shell assignments — `traj_ids="3 7 11"` — so the
format needs no explaining and quoting is explicit. Everything the helper does can be done by hand;
`examples/README.md` says what goes wrong when it is.

## Before the first run

**A Hugging Face token, as an OSMO credential:**

```bash
osmo credential set huggingface_token --type GENERIC --payload-file token=<file-holding-the-token>
```

Every stage that loads a model gets it as `HF_TOKEN`. It is supplied for every model, including ones
whose weights are public, because a token raises Hub rate limits and download throughput — and some
models pull a backbone or tokenizer from a **gated** repository, where an anonymous run fails at load
time with HTTP 403. For those the account behind the token must also have accepted that repository's
terms; a token with the right scope still gets 403 until someone clicks through.

The credential's name is a property of your deployment, not of any model. If yours is called something
else, set `hf_token_credential=<name>`; if you have none, set it empty and expect gated models to fail.

**An S3 bucket** the workflow role can read and write, and **a dataset** in LeRobot format.

**A pool with a GPU platform**, plus a robot platform if you intend to run the hil stage.

| Stage | Needs a GPU | Needs the robot | Publishes |
|---|:---:|:---:|---|
| preflight | no | no | nothing — it echoes the resolved configuration and fails early |
| prepare | model's choice | no | a converted dataset, cached per model and dataset name |
| train | yes | no | checkpoints, one prefix per run, each with a manifest |
| eval | yes | no | metrics, plots, smoothness — **the stop/go gate** |
| hil | yes | **yes** | closed-loop results, trace, latency, lineage |

Stages are groups, and a group runs only if its flag is true. Chaining is native: a later stage takes
`inputs: - task:` from an earlier one, so no S3 prefix is pasted between stages. OSMO round-trips a chained
artifact through S3 regardless, so composing is fewer commands and one less chance to paste the wrong
prefix — not less transfer.

## Adding a model

**A model is a directory. The spec never names one**, because Jinja renders inside `files: localpath:`, so
`models/{{ model }}/train.sh` selects per-model code by path. There is no registry and no conditional on
the model anywhere in the spec.

A model directory is eight files — six scripts, one shared Python module, and `model.vars`:

```
install.sh           SOURCED by every stage; $1 is the stage name
prepare.sh           $1 dataset-in   $2 out
train.sh             $1 dataset  $2 out  $3 steps  $4 batch  $5 save-freq  $6 seed
score.sh             $1 ckpt  $2 dataset  $3 out  $4 traj-ids  $5 horizon  $6 steps
serve.sh             $1 ckpt  $2 dataset  $3 port  $4 camera-names
client_transport.sh  SOURCED by the robot task; exports TRANSPORT_MODULE
transport.py         injected into both hil tasks, so the ends agree on the wire; may be empty
model.vars           the runtime image — what the SPEC must know, not what scripts decide
```

**A model's own code is written by its `install.sh`**, with a heredoc, into `$MODEL_FILES`. So the
pipeline declares no Python, and a model written in another language carries nothing it does not
use. Two rules: no Jinja delimiters in any of these files, because OSMO renders them as templates
(`submit.sh` refuses them); and `--dry-run` does not render payloads, so a template error in a
model's files only appears on a real submit.

A missing file fails at `validate`, naming the path — a good failure, before a node is consumed.

**The principle: the pipeline passes what it owns, the model owns everything else.** The two models here
share no flag vocabulary — one exports environment variables and calls a shell launcher after changing
directory, the other passes dotted CLI flags — which is why the contract is filenames, not flags. Every
script receives every argument whether it uses it or not: ignoring one is free, being unable to see one is
a contract change.

[`docs/CONTRACT.md`](docs/CONTRACT.md) has the obligations behind each file.

## "Model-agnostic" means something narrower than it sounds

The workflow has the contracts it needs to plug in a model. It does **not** mean a model works
irrespective of the robot. A policy server must speak whatever interface the target robot defines, and
**the workflow cannot detect a missing adapter** — it will happily start a server that speaks nothing the
client understands, and the symptom is a client timing out.

For this arm that interface is [`docs/SO101_POLICY_INTERFACE.md`](docs/SO101_POLICY_INTERFACE.md), and writing an adapter for it was ~380 lines for
one of these models and a thin wrapper for the other. That is the cost of adding a model to a *robot*, as
distinct from adding one to the *workflow*.

## The robot half

**The arm moves only with `allow_motion=true`.** The flag reads the way it behaves; an earlier spelling was
a double negative, and that is how a run believed to be powered once moved nothing.

Rehearse in this order:

1. **eval**, on the exact checkpoint you intend to drive with. An under-trained policy and a miswired one
   look identical on hardware.
2. **HIL with `allow_motion` left false.** Everything runs — both tasks gang-schedule, the client resolves
   cameras and opens them, the server validates the observation — with zero writes to the servo bus.
3. **HIL powered**, with a hand on the e-stop.

### Cameras are named, not numbered

`cameras` takes `NAME=SPEC` entries separated by `;`. `NAME` must be the dataset's image-feature suffix,
because that is what the policy server matches on — never position.

The split matters if you swap the environment: the **name** is the client's business, because it goes on
the wire, while the **spec** is interpreted by whatever environment is driving. The selector forms below
are the physical arm's vocabulary; a simulator would define its own, and would report what it resolved
through the same `camera_sources()` that puts the table into `results.json`.

| SPEC | Resolves to |
|---|---|
| `sysfs:*/3-5.1/*` | the capture device whose sysfs path matches — stable per physical port |
| `name:Some Camera*` | the capture device whose v4l2 card name matches — when models differ |
| `/dev/video2`, `0`, `clip.mp4`, a URL | handed to OpenCV untouched |

**Prefer a selector.** `/dev/videoN` is assigned in enumeration order, so a replug or reboot renumbers it
and the run then drives on the wrong camera's frames without complaining. Survey your own bench with:

```bash
for d in /sys/class/video4linux/*; do echo "$d  $(cat $d/name)  index=$(cat $d/index)"; readlink -f $d; done
```

`index` 0 is the capture node; a UVC camera's other node carries only metadata. Two identical cameras
report the same name and an empty serial, so only the sysfs path separates them — and which one is *front*
can still only be settled by looking at a frame.

**No camera `volumeMounts` are declared, and none would help:** a selector resolves on the node while
`volumeMounts` is fixed at submit. What makes resolution work is that the robot task is privileged for the
serial port, and a privileged container receives the host's whole `/dev`.

**The server refuses a camera set the checkpoint was not trained on**, in both directions, before torque.
Not belt-and-braces: one of these servers otherwise *silently drops* an extra camera and the run completes
reporting success, having driven the arm from an input the policy never saw.

### The safety envelope belongs to the arm

Joint margins, per-step caps, deadbands, temperature limits, cooldown and release ramps are all pipeline
variables even though the client defaults every one. They are properties of the *robot*, not of a run and
not of a model — no model chooses any of them — and the point of an e-stop-adjacent limit is that it stays
reachable.

### What the client drives: real arm or simulator

Three files go into the robot task. `eval_so101_bounded.py` is the client, `environment.py` is the
interface it drives through, and `so101_environment.py` is the implementation for a physical arm. To run
against a simulator, ship a different implementation of the interface in place of the third; the client,
the spec, the transport and the policy server are untouched.

Everything device-specific is behind the interface, including the parts that are not obviously about
motion: the serial port, the calibration file, and turning a camera spec into a capture device. The
client names exactly one implementation, on one line, and touches no `/dev` entry and no LeRobot object
itself.

The interface is the *robot's* contract, like the policy interface, and `environment.py` is its
specification. Two rules in it decide whether a substitute is safe:

- **Every member is callable at any time between `connect` and `disconnect`, in any order.** An
  implementation may not need one member to have run before another works. `connect` therefore has to
  establish everything the rest needs — which for the real arm means registering a unit mapping and
  opening the cameras.
- **A query with no answer returns `None`, and `None` never means an empty reading.** `temperatures()`
  returning `None` says "this environment has no notion of heat"; returning `{}` says "I should have been
  able to read it and could not", which the client treats as *still hot*. Same for `torque_state()`: `None`
  is nothing to release, `{}` is an unverified release and reported as a failure to release. Answering
  `{}` where `None` is meant makes a simulator indistinguishable from a servo bus that has gone quiet.

`is_physical()` is recorded in `results.json` and nothing branches on it. It says what produced a result,
not what the client may ask for.

## Reading a result

`hil/<model>/<workflow-id>/` carries `results.json` from the client and `lineage.json` from the pipeline.
The split is deliberate, because neither side knows everything: the client sees numbered mount points
rather than URLs and does not know which model directory was selected, while only the client knows which
physical device each camera name resolved to and whether torque was ever enabled. Together they answer
*what drove the arm* without reference to anyone's shell history.

A checkpoint that arrived from a chained task has no URL — its input is `<workflow-id>/<task-name>/`, which
`lineage.json` records as such rather than leaving blank.

`eval/<model>/<workflow-id>/metrics.json` is the gate's input. **It gates on `complete` alone** — whether
every requested trajectory was actually scored. The error figures are reported and never gated, because
one model's sampling is unseeded: three runs of the same checkpoint over the same trajectories produced
MSE 15.53, 13.70 and 11.49. A numeric threshold on that is a coin toss, and comparing two models from
single runs measures noise. Whether a policy *performs the task* is settled by watching the arm.

## Gotchas that cost a run each

Most of these are what `submit.sh` exists to prevent; they bite when you assemble `--set` by hand.

| Symptom | Cause |
|---|---|
| Submit fails with `NoSuchBucket` against `REPLACE-WITH-…` | `--set` was repeated. It is not additive: only the last group applies. `submit.sh` emits one of each flag |
| `Data 7 is incorrectly formatted` | a space-containing value was word-split by a shell expansion. `submit.sh` passes values as arguments, so spaces and globs are safe |
| A flag you set is absent from preflight's echo | you mistyped the key. `--set` accepts unknown keys silently; `submit.sh` refuses one the spec does not declare, which is the one check the platform cannot do for you |
| A trajectory list arrived as just its first number | `--set` coerces to int/float. `submit.sh` routes non-numeric values to `--set-string` |
| `500 Internal server error: Unknown URI scheme:` at submit | an input URL rendered empty. Defaults are placeholder buckets rather than `""` for this reason |
| A stage fails while the next one completes | a guard that is not an `inputs: - task:` edge is decorative. Failure propagates along edges, never along group order |
| HIL completes and the arm never moved | `allow_motion` was left false. Confirm with `no_motion` in `results.json` |
| HIL aborts on `policy server did not answer within …` | the round trip exceeds `policy_timeout_ms`. Read the warm-up figures the client logs. With pi0-FAST, a barely-trained checkpoint decodes to its token cap and is ~6x slower than a trained one, so check training length before suspecting the network |
| A device is present, correctly owned, and unopenable | the task needs `privileged: true`; the platform's `privileged_allowed` only grants the possibility |
| A group fails to start under a big checkpoint | OSMO validates resources per *task*, not per group, so an oversized task passes validation and then cannot gang-schedule |
