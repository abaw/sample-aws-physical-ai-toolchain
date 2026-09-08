#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Serve a fine-tuned GR00T checkpoint over ZMQ for the closed-loop eval client.

Thin wrapper over upstream's ``gr00t/eval/run_gr00t_server.py``, adding the one thing
stage 04 cannot do without: **an explicit denoising-step count**. Upstream's own
open-loop evaluator refuses ``--denoising-steps`` against a remote server and says to
"set the denoising steps on the server instead", but the server script exposes no such
flag, so the value silently falls back to the checkpoint's config. Denoising steps are
the largest single lever on inference latency, and therefore on whether the control
loop closes at all, so stage 04 must serve at the same value stage 03 evaluated at.

Also logs the served action-chunk length. That number is the ceiling on the client's
``--action-horizon``, and having it in the server log means a horizon rejection can be
diagnosed from either side.

The checkpoint must be a fine-tuned one. The base model has no ``new_embodiment``
entry in its processor and raises at construction — that tag exists only for
fine-tuning.
"""

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Fine-tuned checkpoint directory")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104 - pod-local, one port
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=4,
        help="Diffusion sampling steps. The action head reads this at sampling time.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Skip server-side observation/action validation. Leave it on: it is what "
        "turns a malformed observation into a named error instead of a wrong action.",
    )
    args = parser.parse_args()

    import torch

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer

    if not torch.cuda.is_available():
        # A CPU fallback would load and then serve at several seconds per chunk, which
        # looks like a network problem from the client's side.
        print("ERROR: no CUDA device visible; the policy server needs a GPU", file=sys.stderr)
        return 1

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}", flush=True)

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
        model_path=args.model_path,
        device="cuda",
        strict=not args.no_strict,
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps

    action_config = policy.modality_configs["action"]
    chunk_length = len(action_config.delta_indices)
    print(f"denoising steps: {args.denoising_steps}", flush=True)
    print(f"action chunk length: {chunk_length} (client --action-horizon must not exceed it)", flush=True)
    print(f"action keys: {action_config.modality_keys}", flush=True)
    print(f"state keys: {policy.modality_configs['state'].modality_keys}", flush=True)
    print(f"video keys: {policy.modality_configs['video'].modality_keys}", flush=True)

    # Blocks forever. The client carries `lead: true`, so the group ends when the
    # client exits and this task is torn down with it.
    PolicyServer.start_server(policy=policy, port=args.port, host=args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
