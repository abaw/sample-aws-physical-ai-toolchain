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
passes through. Everything it does can be done by hand; `examples/README.md` says what goes wrong when it
is.

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

A model directory is twelve files. Five are scripts, six are helper slots that **may be empty but must
exist**, and one declares what belongs to the model rather than to a run:

```
install.sh           SOURCED by every stage; $1 is the stage name
prepare.sh           $1 dataset-in   $2 out
train.sh             $1 dataset  $2 out  $3 steps  $4 batch  $5 save-freq  $6 seed
score.sh             $1 ckpt  $2 dataset  $3 out  $4 traj-ids  $5 horizon  $6 steps
serve.sh             $1 ckpt  $2 dataset  $3 port  $4 camera-names
client_transport.sh  SOURCED by the robot task; exports TRANSPORT_MODULE
common.py            every stage        may be empty
prepare_lib.py       prepare            may be empty
train_lib.py         train              may be empty
score_lib.py         eval               may be empty
serve_lib.py         hil                may be empty
transport.py         both hil tasks     may be empty
model.vars           runtime image, credentials — anything the SPEC must know
```

**Why the slots are mandatory rather than optional.** A missing `localpath` fails at `validate`, naming the
path — a good failure, before a node is consumed. An empty one validates fine. So a model that needs no
helper ships an empty file and pays nothing, and nobody has to write a conditional. Injecting a whole
directory would be simpler but is not possible: a directory `localpath` fails with `IsADirectoryError`.

**The principle: the pipeline passes what it owns, the model owns everything else.** The two models here
share no flag vocabulary — one exports environment variables and calls a shell launcher after changing
directory, the other passes dotted CLI flags — which is why the contract is filenames, not flags. Every
script receives every argument whether it uses it or not: ignoring one is free, being unable to see one is
a contract change.

`docs/CONTRACT.md` in the tracker has the obligations behind each file.

## "Model-agnostic" means something narrower than it sounds

The workflow has the contracts it needs to plug in a model. It does **not** mean a model works
irrespective of the robot. A policy server must speak whatever interface the target robot defines, and
**the workflow cannot detect a missing adapter** — it will happily start a server that speaks nothing the
client understands, and the symptom is a client timing out.

For this arm that interface is `eval_so101_bounded.py`'s, and writing an adapter for it was ~380 lines for
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
| A device is present, correctly owned, and unopenable | the task needs `privileged: true`; the platform's `privileged_allowed` only grants the possibility |
| A group fails to start under a big checkpoint | OSMO validates resources per *task*, not per group, so an oversized task passes validation and then cannot gang-schedule |
