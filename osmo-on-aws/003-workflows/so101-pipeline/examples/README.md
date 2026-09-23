# Ready-to-run variable sets

Pick one and hand it to `submit.sh`:

```bash
../submit.sh --example full-chain --model gr00t-n17 \
    storage_url=s3://<your-bucket>/so101 \
    source_dataset_url=s3://<your-bucket>/raw/<name>/ \
    dataset_name=<name> \
    lang_instruction="<the task text your dataset was annotated with>"
```

Add `--dry-run` to see the rendered spec without consuming a node. The helper prints the exact
`osmo workflow submit` it runs, so nothing is hidden and the command can be copied if the helper is ever
in the way.

Everything is `key=value`, in any order, and a value given on the command line beats the same key in a
file. Spaces and globs are safe, because these are ordinary shell arguments.

## The files are shell

```sh
run_prepare="true"
traj_ids="3 7 11"
exec_timeout="24h"
```

Shell assignment syntax, because it is a format every user already knows and it makes quoting explicit
rather than a rule buried in a README. Comments, blank lines and `$HOME` all behave as you would expect.

**They are therefore executed, not parsed.** That is how `.env`-style config is conventionally handled and
is fine for a file in this repo that you edit — but a variables file can run commands, so treat one you
did not write like any other script you would source.

`submit.sh` sources them in an isolated subshell, so a file cannot reach the script's own variables: a
stray `SPEC=` line is reported as an undeclared key rather than silently redirecting your submission to a
different spec. It also diffs the variable names across the source, so a typo *inside a file* is caught
the same as one on the command line.

## What the helper is protecting you from

Assembling this by hand is possible and was what the first version of these docs told you to do. Three
things went wrong with that often enough to be worth a script:

- **`--set` accepts keys nothing references, silently.** `run_trian=false` does nothing, the run proceeds
  on the default, and it looks healthy. The helper checks every key against the spec's own
  `default-values` and refuses an unknown one. **The platform cannot do this for you.**
- **Which flag a value needs is not obvious.** `--set` coerces to int/float, which the spec depends on —
  a storage figure is computed arithmetically — but a trajectory list like `3 7 11` coerced that way
  keeps only the `3`. The helper routes numerics to `--set` and everything else to `--set-string`.
- **Repeating `--set` is not additive.** Only the last group applies, silently, and the run proceeds on
  placeholder defaults. The helper emits exactly one of each flag.

## The model is not part of a composition

None of these files sets `model=`. An example says which **stages** run; which model runs is orthogonal,
and the two should not be bundled — `--model` is a separate argument for that reason.

That is not tidiness. When they were bundled, every example run was a single-model run, so a
model-specific assertion that had leaked into the shared pipeline passed every time, and it survived
until a third model tripped over it.

Every file here leaves the arm **still**. Driving it is `allow_motion=true`, which is deliberately in
none of them.
