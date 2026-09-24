# The SO-101 policy interface

**Contract B: between a robot and a policy server. Defined by the robot, not by the workflow.**

A robot's degrees of freedom, state layout and camera set determine what an observation and an action
*are*. That makes this interface a property of the SO-101 — specifically of its shared client,
`eval_so101_bounded.py`, which is the thing that actually drives the arm and therefore the thing that
gets to say what it sends and expects.

**The workflow does not define, validate, or care about anything in this document.** It launches a
server and a client, hands each the arguments it owns, and gets out of the way. See `CONTRACT.md` for
the part that *is* the workflow's business.

## What this means for "model-agnostic"

A model that satisfies contract A can be prepared, trained and scored by the workflow. **Driving this
arm additionally requires a server that implements this interface.** A model with no such adapter
cannot drive an SO-101 however well it satisfies the workflow's contract, and the workflow cannot
detect the difference — it will start a server that speaks nothing the client understands, and the
symptom is a client that times out waiting for a reply.

Writing that adapter is the real cost of adding a model to a *robot*, as opposed to adding one to the
*workflow*. For the two models that exist it was ~380 lines for one and a thin wrapper for the other.

## Why the client is the authority

Because there is exactly one of it, and it is not a model's to change. Everything below its `ArmDriver`
is hardware safety earned on this arm — per-joint clamping, per-step delta limits, temperature
confirmation with implausible-slew rejection, return-to-start then release, torque read back off the
servos. A second copy that drifts fails by leaving the arm energised, so the client is shared and each
model's server adapts to it rather than the reverse.

## The two shapes

```
observation  {"video":    {"<camera-name>": (1,1,H,W,3) uint8, ...},
              "state":    {"single_arm": (1,1,5) float32, "gripper": (1,1,1) float32},
              "language": {"annotation.human.task_description": [[str]]}}

action       {"single_arm": (1,T,5) float32, "gripper": (1,T,1) float32}
```

Five arm joints plus a gripper, because that is what an SO-101 has. `T` is the chunk length the policy
chose; the client executes `action_horizon` of them and re-plans.

Two leading singleton dimensions on everything — batch and time. They come from the client applying
the same wrapper twice, which is also why the language value arrives **doubly nested** as `[[str]]`.
Unwrap it in a loop rather than with two indexes: a residual list reaches a tokenizer and surfaces as
`'list' object has no attribute 'strip'` from somewhere that says nothing about where it came from.

`float32` is not incidental. A server that validates dtypes and rejects anything else is the good
outcome; the alternative is a silently rescaled state.

## The transport is the model's choice

The bytes are **not** part of this interface. The client loads its transport by name:

```
--transport-module <importable module>
```

That module must provide exactly this much, and nothing about how it moves data is specified:

```python
class PolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int): ...
    def ping(self) -> bool: ...                                  # MUST NOT raise
    def get_action(self, obs: dict) -> tuple[dict, dict]: ...     # (action, info)
    def close(self) -> None: ...

class MsgSerializer:
    @staticmethod
    def to_bytes(obj) -> bytes: ...      # payload-size reporting and a round-trip
    @staticmethod                        # diagnostic of the observation only
    def from_bytes(b: bytes): ...
```

**`ping()` must never raise.** The client polls it bare — `if self.client.ping():` — for up to half an
hour while a server fetches a checkpoint and loads a model, so a timeout there is the normal case and
must read as "not ready yet". Raising ends the run before the server has had a chance to come up. This
has already broken a run.

A model with no reason to differ should reuse the shipped implementation: ZMQ request/reply over
msgpack with a hard no-pickle boundary.

## Cameras: three identities, one of which belongs here

| Identity | Example | Owned by |
|---|---|---|
| how the client opens it | `sysfs:*/3-5.1/*`, or `/dev/video2` | the machine |
| dataset feature key | `observation.images.front` | the recording |
| the model's internal key | `front` | the model |

The first is deliberately not "a device path". A v4l2 node's number depends on plug order, so a camera
is named by a selector resolved against sysfs at run time, with a literal device node, index, file or
URL accepted verbatim; see the pipeline's README. **None of that reaches the wire.** It is how the client acquires
frames; the server only ever sees the name.

**The wire name is the dataset's image-feature suffix.** A dataset's `meta/modality.json` maps between
the last two — `"front": {"original_key": "observation.images.front"}` — and that indirection exists
precisely because they need not match. The two shipped servers already read different ones: one resolves
by the dataset key's last dot-segment, the other by its own `modality_keys`. They agree only because
this recording happens to use matching names.

A server whose internal naming differs maps it internally. Cameras are matched **by name, never by
position**: swapping two produces a confident wrong trajectory that no metric in the pipeline detects,
and the arm faithfully goes where it is told.

**A server must refuse a camera set the checkpoint was not trained on, before torque.** It holds the
checkpoint and the dataset, so it is the only component that can. A two-camera checkpoint cannot
consume a one-camera observation, and a one-camera rig needs a one-camera dataset and checkpoint.

## What is fixed, and what a different robot would change

| | |
|---|---|
| Fixed | six joints, split 5 + 1; two leading singleton dimensions; `float32`; camera names from the dataset |
| Flexible | how many cameras and what they are called; the chunk length `T`; the transport |

A robot with different degrees of freedom needs its own client **and** its own server adapters — the
state split above is not parameterised, deliberately, because the safety behaviour below it was earned
on this hardware and a generic version would either exclude most robots or offer weaker guarantees
under the same name.
