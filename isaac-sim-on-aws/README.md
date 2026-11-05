# Isaac Sim on AWS

Deploy [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) on AWS for physics-accurate robot simulation. Isaac Sim provides photorealistic rendering, accurate physics, and sensor simulation for developing and testing robot policies before deploying to real hardware.

---

## Two Modes of Operation

| Mode | Use Case | Guide |
|------|----------|-------|
| **GUI Mode** | Visual development, debugging, scene building, watching policies run | [GUI Mode Guide →](gui-mode-guide.md) |
| **Headless Mode** | Large-scale training, batch rendering, CI/CD validation | [Headless Mode Guide →](headless-mode-guide.md) |

---

## When to Use Each Mode

**GUI Mode** — deploy a GPU workstation with NICE DCV remote desktop:
- Iterate on RL environments visually
- Debug physics issues (objects clipping, falling through surfaces)
- Watch trained policies execute in real-time
- Build and compose simulation scenes
- Validate camera views and sensor placements

**Headless Mode** — run simulation without rendering:
- Train RL policies at scale (4096+ parallel environments)
- Generate synthetic data (domain randomization)
- Run automated evaluation pipelines
- CI/CD simulation testing

---

## Infrastructure

Both modes use the same Terraform to deploy a GPU workstation:

```bash
cd isaac-sim-on-aws/infra
terraform init
terraform apply -var="aws_region=us-east-2"
```

| Resource | Description |
|----------|-------------|
| EC2 Instance | g6e.4xlarge (L40S GPU, 48 GB VRAM) |
| AMI | NVIDIA Isaac Sim Marketplace AMI (pre-installed) |
| EBS | 512 GB gp3 |
| Security Group | DCV (8443) + SSH (22) |
| IAM Role | SSM + ECR access |

---

## Prerequisites

- Foundation infrastructure deployed
- [Isaac Sim Marketplace AMI subscription](https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw) (free, one-time per account)
- EC2 GPU quota for g6e.4xlarge
- NVIDIA NGC API key (for pulling container images)

---

## Cost

| Item | Cost | Notes |
|------|------|-------|
| g6e.4xlarge (running) | ~$1.86/hr | Stop when not in use |
| EBS 512 GB | ~$40/month | Persists when instance is stopped |

**Always stop the instance when done** — it bills per second while running.

---

## Troubleshooting

Common failures when deploying and using the workstation. For problems inside Isaac Sim or the container, each mode guide has its own table: [GUI Mode Guide → Troubleshooting](gui-mode-guide.md#troubleshooting), [Headless Mode Guide → Troubleshooting](headless-mode-guide.md#troubleshooting).

### Deploy (both modes)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `terraform apply` stops and prompts for `var.ami_id` | `ami_id` has no default in `infra/variables.tf` | Pass it: `terraform apply -var="aws_region=us-east-2" -var="ami_id=<AMI_ID>"`, using the AMI ID from the Marketplace listing's **Launch** tab for your region |
| `OptInRequired` or "you are not subscribed to this service" on instance launch | The account has not subscribed to the Isaac Sim Marketplace AMI | Subscribe once per account at the [listing](https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw) (**Continue to Subscribe → Accept Terms**), wait for it to show **Active**, then re-run `terraform apply` |
| `InvalidAMIID.NotFound` or `InvalidAMIID.Malformed` | The AMI ID belongs to another region, or to an older Isaac Sim release | AMI IDs are regional and change with each release; copy the current ID for `aws_region` from the listing's **Launch** tab |
| `VcpuLimitExceeded` | The account's **Running On-Demand G and VT instances** quota is below the 16 vCPUs a g6e.4xlarge needs (new accounts often have 0) | Request an increase in Service Quotas (EC2 → *Running On-Demand G and VT instances*) in the deploy region; approval can take hours to days |
| `InsufficientInstanceCapacity` or `Unsupported` for g6e.4xlarge | No L40S capacity in the chosen Availability Zone, or g6e is not offered there | Retry later, or set `-var="subnet_id=<SUBNET_ID>"` to a public subnet in an AZ or region that offers g6e |
| Launch fails with a volume-size error | `volume_size` was set below what the AMI requires | Keep `volume_size` at the default `512` (GB) or higher |
| `VPCIdNotSpecified` or no default VPC found | `subnet_id` is empty, so the instance targets the default VPC, and the region has none | Pass `-var="subnet_id=<SUBNET_ID>"` for a public subnet (route to an internet gateway, auto-assign public IP on) |
| `EntityAlreadyExists` for the IAM role or instance profile | IAM names are global: a workstation with the same `project_name` and `environment` already exists in another region, or a failed destroy left them behind | Deploy with a different `-var="environment=<name>"`, or destroy the other copy first |
| `public_ip` / `dcv_url` output is empty | The subnet does not auto-assign public IPv4 addresses | Use a public subnet (see `subnet_id` above); DCV needs a public IP |

### Connecting (both modes)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `aws ssm start-session` returns `TargetNotConnected` | The SSM agent has not registered yet, the instance has no outbound internet route, or the Session Manager plugin is missing locally | Wait 2-3 minutes after launch or reboot; check the instance shows **Online** in Systems Manager → Fleet Manager; confirm the subnet routes to an internet gateway; install the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) |
| SSH or `https://<IP>:8443` times out | Your IP is outside `allowed_cidrs`, or you are using the IP from before a stop/start | Re-fetch the public IP (it changes on every start, see [Start / Stop](gui-mode-guide.md#start--stop-the-workstation)); if you set `allowed_cidrs`, make sure it covers your current address |

### GUI mode

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| DCV shows a black screen, or `nvidia-smi` fails | The AMI needs one reboot after first launch to initialise the GPU display | `sudo reboot` and wait about 2 minutes ([GUI guide step 3](gui-mode-guide.md#step-3-reboot-first-time-only)) |
| DCV login rejects the user | Wrong user name, or no password set | Log in as `ubuntu` after `sudo passwd ubuntu` over SSM |
| Isaac Lab container segfaults in `librtx.scenedb.plugin.so` when rendering | The ECR `physical-ai/isaac-lab:latest` (2.1.0) and `2.3.2` images are incompatible with the AMI's NVIDIA driver for rendering | Use `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` for GUI work (see the [container compatibility matrix](gui-mode-guide.md#container-compatibility-matrix)) |
| `docker pull nvcr.io/...` returns `unauthorized` | Docker is not logged in to NGC | `docker login nvcr.io --username '$oauthtoken'` and enter your NGC API key as the password (keep the single quotes: the user name is literally `$oauthtoken`) |
| Isaac Sim window does not open from the container | The container has no access to the DCV display | Run `xhost +` in the DCV terminal and pass `-e DISPLAY --network=host`, as in the [GUI guide](gui-mode-guide.md#launch-the-container-with-display-passthrough) |

### Headless mode

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `docker pull` from ECR returns `no basic auth credentials` | The ECR login expired (tokens last 12 hours) or was for another region | Re-run the `aws ecr get-login-password ... \| docker login ...` step from the [headless guide prerequisites](headless-mode-guide.md#prerequisites) for the region the image lives in |
| ECR pull returns `not found` or `repository does not exist` | The `physical-ai/isaac-lab` repository or image is not in this account and region | Deploy the [foundation](../foundation/README.md) stack and push the Isaac Lab image in the same region as the workstation |
| `aws s3 cp` returns `AccessDenied` | The workstation role grants only `s3:GetObject` and `s3:ListBucket` on `physical-ai-*` buckets | Read checkpoints from a `physical-ai-*` bucket, or copy them there first; the role cannot write to S3 |
| Container crashes with `Bus error` or is OOM-killed | Shared memory too small for Isaac Sim | Keep `--shm-size=60g` (or higher) on every `docker run` |
| `Segmentation fault` with `--video` | Video rendering in the headless container is a known failure | Drop `--video` and keep `--headless`; watch the policy in GUI mode instead |
