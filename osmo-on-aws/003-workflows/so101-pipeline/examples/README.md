# Ready-to-run variable sets

Each file is a list of `key=value` lines. Expand one into a single `--set`:

```bash
osmo workflow submit ../pipeline.yaml \
  --set $(grep -v '^#' full-chain.vars | grep -v '^$' | tr '\n' ' ') \
        storage_url=s3://<your-bucket>/so101 \
        $(grep -v '^#' ../models/gr00t-n17/model.vars | grep -v '^$' | tr '\n' ' ')
```

**One `--set`, always.** The flag takes many values but is not additive: repeating it keeps only the last
group, silently, and the run proceeds on placeholder defaults. Preflight echoes every resolved value for
exactly this reason — if something you passed is missing from that list, you mistyped the key.

`osmo workflow submit ... --dry-run` prints the rendered spec without consuming a node.

## These files hold only shell-safe values, and that is a rule not an accident

The expansion above is word-splitting, so **a value containing a space cannot live in one of these
files.** `traj_ids=3 7 11` becomes three arguments and the submit fails with
`Data 7 is incorrectly formatted` — which names the fragment, not the variable, so it reads like a
mystery. A value containing `*` is worse: the shell globs it against your working directory before OSMO
ever sees it.

So anything with a space or a shell metacharacter is passed on the command line, quoted:

```bash
osmo workflow submit ../pipeline.yaml \
  --set $(grep -v '^#' full-chain.vars | grep -v '^$' | tr '\n' ' ') \
        storage_url=s3://<your-bucket>/so101 \
        source_dataset_url=s3://<your-bucket>/raw/<name>/ \
        dataset_name=<name> \
  --set-string traj_ids="3 7 11" \
               lang_instruction="<the task text your dataset was annotated with>" \
               cameras="front=sysfs:*/3-5.1/*;wrist=sysfs:*/3-1/*"
```

`--set-string` for those three: it skips the int/float coercion `--set` applies, which matters for a value
like `3 7 11` that looks numeric at the front.

Every file here leaves the arm **still**. Driving it is `allow_motion=true`, which is deliberately not in
any of them.
