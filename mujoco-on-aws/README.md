# MuJoCo on AWS

> **Status: Preview.** The infrastructure, image and launcher are in place but have not yet been validated end to end on hardware. The status moves to Available once a Batch job and a headless playback have run in a dev account.

Run [MuJoCo](https://mujoco.org/) on AWS for GPU-parallel robot RL. MuJoCo is the open-source physics engine most research RL code targets. Its JAX port, [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html), steps thousands of environments in parallel on one GPU. The image trains with Brax PPO on [MuJoCo Playground](https://playground.mujoco.org/) envs and renders rollouts to MP4 with no display.

---

## Two Modes of Operation

| Mode | Use Case | Guide |
|------|----------|-------|
| **Headless Simulation** | One GPU instance for interactive training, policy playback from a checkpoint, and offscreen MP4 rendering | [Headless Simulation Guide →](headless-sim-guide.md) |
| **Batch RL Training** | An AWS Batch queue that runs single-GPU MJX PPO jobs and writes checkpoints to S3, scaling to zero when idle | [Batch RL Training Guide →](batch-rl-training-guide.md) |

---

## When to Use Each Mode

**Headless Simulation** — one GPU instance you reach over SSM Session Manager:
- Try a Playground env or a training config before you queue it
- Play back a trained checkpoint and render it to MP4
- Debug the image or a training run interactively

**Batch RL Training** — submit jobs and walk away:
- Run training jobs without keeping an instance up
- Run several seeds or envs side by side (one GPU per job)
- Keep checkpoints and a rollout MP4 per job in the foundation checkpoints bucket

---

## How It Differs From Isaac

| | MuJoCo | Isaac Lab / Isaac Sim |
|-|--------|-----------------------|
| Container base | Public NGC CUDA image (anonymous pull), packages from PyPI | NGC image (`nvcr.io/...`) |
| NGC API key | Not needed | Needed to build the image |
| Image size | A few GB | ~16 GB |
| Batch job | Single node, 1 GPU, job definition created by Terraform | Multi-node job definition (CDK), or registered by hand (Terraform) |
| Batch idle cost | Scales to zero (`min_vcpus = 0`) | Keeps 16 vCPUs warm while Batch is enabled |
| Instance access | SSM Session Manager only, no inbound ports | DCV (8443) and SSH (22) |
| GUI | None (renders to MP4) | NICE DCV workstation |

---

## Infrastructure

Both modes deploy from one Terraform root, each behind its own flag:

```bash
cd mujoco-on-aws/infra
terraform init
terraform apply -var="aws_region=us-east-2" -var="enable_batch=true" -var="enable_sim=true"
```

Set only the flag for the mode you want. The region must match the one foundation is deployed in. If your deploy role may only create roles that carry a permissions boundary, add `-var="permissions_boundary_arn=arn:aws:iam::<ACCOUNT_ID>:policy/physical-ai-mujoco-boundary"` to every `apply`: an apply without it removes the boundary from the roles.

| Resource | Created when | Description |
|----------|--------------|-------------|
| CodeBuild project | Always | `physical-ai-dev-mujoco-build`, builds `physical-ai/mujoco:latest` with no NGC key |
| Batch compute environment | `enable_batch` | g6e.4xlarge (1 L40S GPU), `min_vcpus = 0`, foundation private subnets |
| Batch job queue + job definition | `enable_batch` | `physical-ai-dev-mujoco-queue`, `physical-ai-dev-mujoco-train` (1 GPU, `batch-train`) |
| EC2 instance | `enable_sim` | g6e.xlarge on the Deep Learning Base GPU AMI, 100 GB encrypted gp3, IMDSv2 |
| Security groups | Per mode | Egress only, no inbound rules |
| IAM roles | Per mode | Batch uses its service-linked role `AWSServiceRoleForBatch` (if `aws iam get-role --role-name AWSServiceRoleForBatch` fails, create it once with `aws iam create-service-linked-role --aws-service-name batch.amazonaws.com` before the apply). Batch job: read/write on the checkpoints and models buckets. Instance: SSM + ECR read-only + S3 read on those buckets |

MuJoCo creates no VPC, buckets or ECR repository of its own: it reads them from the [foundation](../foundation/README.md) SSM parameters. Terraform files: [`infra/main.tf`](infra/main.tf), [`infra/batch.tf`](infra/batch.tf), [`infra/ec2.tf`](infra/ec2.tf).

The Batch queue and headless instance ship as Terraform only, and they need the Terraform foundation. The CDK Foundation stack also builds the image (its `MujocoBuild` project), but it publishes none of the `/physical-ai/*` SSM parameters this root reads ([foundation → Deploy with CDK](../foundation/README.md#deploy-with-cdk)), so `mujoco-on-aws/infra` cannot deploy on top of it.

---

## Prerequisites

- Foundation infrastructure deployed ([foundation](../foundation/README.md)), which creates the `physical-ai/mujoco` ECR repository
- EC2 GPU quota: **Running On-Demand G and VT instances** of at least 16 vCPUs for Batch (g6e.4xlarge) or 4 for the headless instance (g6e.xlarge)
- Terraform >= 1.5 and the AWS CLI
- For headless mode: the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) for the AWS CLI

No NGC account or Marketplace subscription is needed.

---

## Cost

| Item | Cost | Notes |
|------|------|-------|
| Batch compute (g6e.4xlarge) | ~$3.00/hr | Only while a job runs; the compute environment scales to zero when the queue is empty |
| Headless instance (g6e.xlarge) | ~$1.86/hr | Stop when not in use |
| EBS 100 GB (headless instance) | ~$8/month | Persists when the instance is stopped |
| CodeBuild image build | ~$0.02 per build-minute | `BUILD_GENERAL1_LARGE`, billed only while a build runs |

On-demand prices for us-east-2. **Stop the headless instance when done** — it bills per second while running.

---

## Code

| Path | What it is |
|------|------------|
| [`containers/mujoco/Dockerfile`](../containers/mujoco/Dockerfile) | The image: pinned `mujoco`, `mujoco-mjx`, `playground`, `brax`, `jax[cuda12]`, `MUJOCO_GL=egl` |
| [`containers/mujoco/train_entrypoint.py`](../containers/mujoco/train_entrypoint.py) | Brax PPO training, checkpoints, rollout MP4, S3 upload |
| [`training/scripts/launch_mujoco_batch.py`](../training/scripts/launch_mujoco_batch.py) | Submits a Batch training job (`--dry-run` previews it) |
