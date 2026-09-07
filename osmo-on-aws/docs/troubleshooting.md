# OSMO on AWS - Troubleshooting Guide

## Common Issues

### 1. Terraform Errors

#### "Error: creating EKS Cluster"

**Symptoms:**
```
Error: creating EKS Cluster (osmo-cluster): operation error EKS: CreateCluster
```

**Causes & Solutions:**
- IAM permissions insufficient → Ensure IAM user/role has EKS permissions
- Subnet configuration invalid → Check VPC CIDR ranges don't overlap
- Service quota exceeded → Request quota increase for EKS clusters

#### "Error: Provider produced inconsistent final plan"

**Solution:** Run `terraform refresh` then `terraform plan` again.

### 2. EKS Connection Issues

#### "Unable to connect to the server"

**Symptoms:**
```
Unable to connect to the server: dial tcp: lookup ... on ...: no such host
```

**Causes & Solutions:**
1. Kubeconfig not configured:
   ```bash
   aws eks update-kubeconfig --region <region> --name <cluster-name>
   ```

2. VPN required (private endpoint only):
   - Connect to VPN if cluster has `cluster_endpoint_public_access = false`

3. IAM permissions:
   - Ensure your IAM identity is in the cluster's access entries
   ```bash
   aws eks list-access-entries --cluster-name <cluster-name>
   ```

### 3. Pod Issues

#### Pods stuck in "Pending"

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n osmo
```

**Common causes:**

1. **Insufficient resources:**
   ```
   0/3 nodes are available: 3 Insufficient cpu
   ```
   → Scale up node group or reduce resource requests

2. **Node selector/taint issues:**
   ```
   0/3 nodes are available: 3 node(s) didn't match Pod's node affinity
   ```
   → Check node labels and pod tolerations

3. **PVC pending:**
   ```
   pod has unbound immediate PersistentVolumeClaims
   ```
   → Check storage class and EBS CSI driver

#### Pods stuck in "ImagePullBackOff"

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n osmo | grep -A5 "Events:"
```

**Solutions:**
1. NGC registry secret missing:
   ```bash
   kubectl create secret docker-registry ngc-registry-secret \
     -n osmo \
     --docker-server=nvcr.io \
     --docker-username='$oauthtoken' \
     --docker-password="<NGC_API_KEY>"
   ```

2. ECR authentication:
   ```bash
   # IRSA should handle this, but verify the role
   kubectl describe serviceaccount osmo-service -n osmo
   ```

#### Pods in "CrashLoopBackOff"

**Diagnosis:**
```bash
kubectl logs <pod-name> -n osmo --previous
```

**Common causes:**

1. **Database connection failed:**
   - Check RDS security group allows traffic from EKS
   - Verify database credentials in secret

2. **Redis connection failed:**
   - Check ElastiCache security group
   - Verify auth token in secret

3. **Missing configuration:**
   - Check ConfigMaps and Secrets are created
   - Verify environment variables

### 4. Database Connectivity

#### Cannot connect to RDS

**Test from a pod:**
```bash
kubectl run -it --rm psql-test --image=postgres:15 \
  --restart=Never -- \
  psql -h <rds-endpoint> -U osmo_admin -d osmo
```

**Checklist:**
- [ ] RDS security group allows port 5432 from EKS node security group
- [ ] Pod is in a private subnet that can reach RDS
- [ ] Password secret is correct

### 5. Redis Connectivity

#### Cannot connect to ElastiCache

**Test from a pod:**
```bash
kubectl run -it --rm redis-test --image=redis:7 \
  --restart=Never -- \
  redis-cli -h <redis-endpoint> -p 6379 --tls -a <auth-token> ping
```

**Checklist:**
- [ ] ElastiCache security group allows port 6379 from EKS
- [ ] TLS is enabled (required for auth tokens)
- [ ] Auth token secret is correct

### 6. Load Balancer Issues

#### ALB not provisioning

**Diagnosis:**
```bash
kubectl describe ingress osmo-service-ingress -n osmo
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
```

**Common causes:**

1. **Subnet tags missing:**
   - Public subnets need: `kubernetes.io/role/elb=1`
   - Private subnets need: `kubernetes.io/role/internal-elb=1`

2. **IAM permissions:**
   - Check AWS LB Controller IRSA role has correct policies

3. **Ingress class mismatch:**
   - Verify `ingressClassName: alb` in Ingress resource

### 7. GPU Issues

#### GPUs not detected

**Diagnosis:**
```bash
# Check GPU operator
kubectl get pods -n gpu-operator

# Check node labels
kubectl get nodes -l nvidia.com/gpu.present=true

# Check device plugin
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
```

**Solutions:**

1. **Driver not loaded:**
   - EKS GPU AMI should have drivers pre-installed
   - Check AMI type: `AL2_x86_64_GPU`

2. **Device plugin not running:**
   - Check GPU operator deployment
   - Verify node taints and tolerations

3. **Wrong instance type:**
   - Ensure nodes are GPU instance types (g4dn, g5, p3, p4d, etc.)

#### GPU Operator pods crash on containerd 2.x AMIs

**Symptom:** After a fresh deploy, pods in `gpu-operator` (notably
`nvidia-container-toolkit-daemonset` and `nvidia-operator-validator`) crash-loop,
and GPU nodes never advertise `nvidia.com/gpu` capacity.

**Cause:** GPU Operator **v24.9.0** ships a container-toolkit that is incompatible
with the **containerd 2.x** runtime on current EKS GPU AMIs — the toolkit writes a
`config.toml` in a schema that containerd 2.x rejects, so the runtime never gets
NVIDIA support wired in.

**Solution:** use GPU Operator **v25.10.1** or later (now the default in
`config/deployment-config.yaml.example` and `defaults.conf`). If you overrode it,
set `dependencies.gpu_operator_version: "v25.10.1"` (or `export
GPU_OPERATOR_VERSION=v25.10.1`) and re-run `02-deploy-gpu-infrastructure.sh`.

#### Node is `Ready` but advertises `nvidia.com/gpu: 0`, driver pod `CrashLoopBackOff`

**Symptom:** a GPU node is `Ready` and schedulable, but reports no GPUs.
`nvidia-driver-daemonset` crash-loops and everything downstream of it is stuck
behind `nvidia-operator-validator`:

```
nvidia-driver-daemonset-xxxxx      0/1  CrashLoopBackOff
nvidia-operator-validator-xxxxx    0/1  Init:0/4
nvidia-device-plugin-daemonset-x   0/1  Init:0/1
```

Note that nothing alarms: the node stays `Ready`, so GPU pods sit `Pending`
forever, and a resource spec that does *not* request `nvidia.com/gpu` will
schedule onto the node and run with no GPU at all.

**This looks identical to the containerd 2.x failure above.** The only way to tell
them apart is to read the driver log — always do this first:

```bash
# The first line names the kernel the driver is trying to build against
kubectl -n gpu-operator logs -l app=nvidia-driver-daemonset --tail=50 | head -20

# Compare with the kernel the node is actually running
kubectl get nodes -o custom-columns='NAME:.metadata.name,KERNEL:.status.nodeInfo.kernelVersion'
```

If the log ends in `The nvidia-uvm kernel module failed to build`, this entry
applies. If instead the toolkit is rejecting `config.toml`, see the containerd 2.x
entry above.

**Cause:** on Ubuntu GPU nodes, the driver container compiles its kernel modules
against the node's *running* kernel. Ubuntu's `unattended-upgrades` will upgrade
the `linux-image-aws` metapackage across a whole kernel series (observed:
`6.17` → `7.0`), and the next reboot lands the node on a kernel the pinned driver
has no `nvidia-uvm` support for. Pinning `gpu_ami_id` fixes only the *initial*
kernel, so it gives no protection once the node has been running for a while.

```bash
# On-node confirmation (via SSM):
uname -r                    # running kernel — newer than the AMI's?
apt-mark showhold           # should list the kernel metapackages
grep -h linux /var/log/unattended-upgrades/unattended-upgrades.log*
```

**Solution:** GPU node groups now hold their kernel metapackages from user-data
(`001-iac/modules/eks/node-groups.tf`), so a node launched from that launch
template cannot drift. To recover a node that has *already* drifted, **replace it** —
cycle the GPU node group rather than trying to roll the kernel back:

```bash
# Force a rolling replacement onto the pinned AMI
aws eks update-nodegroup-version --cluster-name <cluster> --nodegroup-name <gpu-ng>
```

Two traps when recovering:

- **Booting back into the old kernel usually is not possible.** `unattended-upgrades`
  autoremoves superseded kernels, so the AMI's original kernel is typically already
  gone from `/boot` and from the GRUB menu — check `dpkg -l | grep <old-version>`
  for `rc` (removed) status before planning a rollback.
- **`ls /lib/modules` over-reports.** An autoremoved kernel leaves a directory
  containing only `modules.*` metadata and no `kernel/` subdirectory. It looks
  present but is not bootable; trust `/boot` and `grep menuentry /boot/grub/grub.cfg`
  instead.

Because the kernel is held, refreshing `gpu_ami_id` is now the only way GPU nodes
receive kernel security patches — re-pin it periodically and cycle the node group,
confirming the driver still builds against the new AMI's kernel.

#### GPU nodes fail to launch — `VcpuLimitExceeded` / `InsufficientInstanceCapacity`

**Symptom:** the GPU node group never reaches its desired count; the backing EC2
Auto Scaling group shows failed scaling activities.

**Diagnosis:**
```bash
# Inspect the most recent scaling activities for the GPU node group's ASG
aws autoscaling describe-scaling-activities \
  --auto-scaling-group-name <gpu-node-group-asg> --max-items 5
```

**Causes & solutions:**
- **`VcpuLimitExceeded`** — the account's On-Demand GPU vCPU quota is too low. For
  the g5/g6 families the quota is **L-DB2E81BA** ("Running On-Demand G and VT
  instances"); the P family (p4d/p5) has a separate quota code. Request an increase
  in Service Quotas for the deployment region, then retry.
- **`InsufficientInstanceCapacity`** — AWS has no capacity for that instance type in
  the chosen AZ. Try another AZ or instance type, or back the node group with an
  On-Demand Capacity Reservation.

### 8. OSMO-Specific Issues

#### Backend not connecting to service

**Diagnosis:**
```bash
kubectl logs -n osmo -l app.kubernetes.io/name=osmo-backend-listener
```

**Checklist:**
- [ ] Service URL is correct and reachable
- [ ] Backend operator token is valid
- [ ] WebSocket connection is not blocked by security groups

#### Workflow stuck in "Pending"

**Diagnosis:**
```bash
kubectl get pods -n osmo-workflows
kubectl describe pod <workflow-pod> -n osmo-workflows
```

**Common causes:**
- GPU nodes not available (check autoscaler)
- Resource quota exceeded
- PVC cannot be provisioned
- GPU node root disk too small for large images (Isaac Sim / Cosmos) → see `gpu_node_volume_size` (default 500 GiB) in `terraform.tfvars`; symptom is `Evicted: node was low on resource: ephemeral-storage` or `ErrImagePull: no space left on device`.

#### Dataset write fails with S3 `AccessDenied` (`GetBucketLocation`)

**Symptom:** a workflow task fails during "Validating WRITE access for dataset output" with:
```
User: arn:aws:sts::<acct>:assumed-role/<...>-gpu-eks-node-group/<i-...> is not authorized
to perform: s3:GetBucketLocation ...
```

**Cause:** `osmo-ctrl` in the task pod fell back to the **EKS node IAM role** instead of using IRSA. In the 6.3 model, workflow task pods must run under the `osmo-workflow` ServiceAccount (annotated with the `osmo_workflow_role_arn`).

**Solutions:**
1. Verify the SA exists and is annotated:
   ```bash
   kubectl get sa osmo-workflow -n osmo-workflows -o yaml | grep role-arn
   ```
   If missing, re-run `05-deploy-osmo-backend.sh` (it creates + annotates the SA).
2. Verify the pod template pins the SA — `services.configs.podTemplates.gpu_tolerations.spec.serviceAccountName: osmo-workflow` in `values/osmo-control-plane.yaml`. Re-run `04` after changes.
3. Confirm the IRSA trust policy subject matches `osmo-workflows:osmo-workflow` (see `001-iac/modules/eks/irsa.tf`).
4. Ensure the `osmo-workflows` namespace allows HTTPS egress to STS + S3 (NetworkPolicy `allow-workflow-aws-egress`).

#### `s3://` workflow I/O fails with `ValueError: Invalid endpoint:`

**Symptom:** a workflow that reads/writes `s3://…` dataset or artifact paths fails
early with an error resembling:
```
ValueError: Invalid endpoint: https://s3..amazonaws.com
```
(note the empty region between the dots).

**Cause:** upstream bug NVIDIA/OSMO#1067 — on the IRSA credential path `osmo-ctrl`
constructs the S3 endpoint without a region, producing an invalid host. It surfaces
only when the task authenticates via **IRSA**, which is the required model on AWS
(company policy prohibits long-lived data-bucket credentials).

**Solution (workaround):** set `OSMO_SKIP_DATA_AUTH=1` on the task pod template.
This repo already applies it in `values/osmo-control-plane.yaml` under
`services.configs.podTemplates.default_ctrl`; if you customized pod templates,
re-add the env var and re-run `04-deploy-osmo-control-plane.sh`. Remove the
workaround once #1067 is fixed upstream.

#### Config change via CLI/API returns HTTP 409

**Cause:** 6.3 runs in **ConfigMap configuration mode** (`services.configs.enabled: true`), which makes the config API read-only.

**Solution:** edit the declarative values in `values/osmo-control-plane.yaml` (`services.configs.*`) and re-run `04-deploy-osmo-control-plane.sh`. A malformed config crash-loops new pods — check `kubectl describe configmap osmo-service-configs -n osmo` for a `ConfigMapReloadFailed` event.

#### `kubectl logs`/`exec`/`port-forward` fail with "Authorization error ... nodes/proxy"

**Cause:** the API server's kubelet client lacks RBAC for the `nodes/proxy` subresource (cluster-wide), which breaks logs/exec/port-forward — and therefore the bootstrap scripts that port-forward to `osmo-service`.

**Solution:** this is a cluster-level kubelet authorization issue (often from a custom node AMI bootstrapping kubelet authz differently). Verify a healthy fresh EKS cluster has the default apiserver→kubelet RBAC; if using a custom GPU AMI, ensure its bootstrap configures `--authorization-mode=Webhook` / `--authentication-token-webhook` consistently with the EKS-optimized system nodes.

### 9. Authentication / Login Issues

#### Forced to re-login roughly every hour

**Cause:** the `osmo` realm is created (by `03-deploy-keycloak.sh`) with short
lifetimes — `accessTokenLifespan` 3600s (1h) and `ssoSessionIdleTimeout` 1800s
(30m idle). When the session expires, oauth2-proxy bounces the browser back to
Keycloak for re-authentication.

**Solution:** raise the realm's `accessTokenLifespan` / `ssoSessionIdleTimeout`
(Keycloak Admin → Realm settings → Tokens/Sessions) to match your security posture,
then have users re-authenticate once. No OSMO redeploy is needed. To change the
baked-in defaults for future deploys, edit the realm-creation body in
`03-deploy-keycloak.sh`.

#### `osmo` CLI returns 401 / token expired

**Causes & solutions:**
- **Expired credential** — re-run `osmo login` to refresh. The device-login code
  itself is short-lived (`oauth2DeviceCodeLifespan` 600s / 10m), so complete the
  browser step promptly.
- **CLI / API version skew** — a CLI built against a different OSMO API version can
  fail auth or reject responses. Match the CLI to the deployed control plane
  (`osmo.version` in `config/deployment-config.yaml`, currently `6.3.1`).

### 10. Hybrid Node Issues

#### `terraform plan` wants to replace the cluster after enabling hybrid nodes

**Symptoms:**
```
# module.eks.module.eks.aws_eks_cluster.this[0] must be replaced
      + remote_network_config { # forces replacement
```

**Cause:** the EKS API supports adding, changing and removing remote networks on a
live cluster, but the Terraform provider models the block as create-only.

**Solution:** do not apply. Set the configuration through the API, wait for it to
finish, then enable `should_deploy_hybrid_nodes` — the refreshed state matches the
configuration, so nothing is left to replace. Commands in
`docs/deployment-modes.md`.

#### "CIDR overlaps with control plane VPC CIDR"

**Symptoms:**
```
InvalidParameterException: Invalid remote node network: CIDR 172.16.90.0/29
overlaps with control plane VPC CIDR 172.16.0.0/16
```

**Cause:** besides your VPC and the Service CIDR, the hybrid networks must also
avoid the VPC EKS runs the control plane in. That CIDR is not exposed by any API,
so nothing can validate it ahead of time.

**Solution:** take the CIDR from the error message and move `hybrid_node_cidrs` /
`hybrid_pod_cidrs` out of it. The repo defaults avoid `172.16.0.0/12` for this
reason.

#### "must be within 10.0.0.0/8, 172.16.0.0/12, or 192.168.0.0/16"

**Cause:** a provider-side check on `remote_network_config`, stricter than the EKS
API — CGNAT (`100.64.0.0/10`) is documented by AWS and accepted by the API, but
rejected here. It appears at the end of `plan`, after the rest of the plan has
already rendered.

**Solution:** use RFC-1918 space. Terraform's own guard catches this earlier with a
clearer message, so seeing the provider error means the CIDRs were changed
somewhere that bypasses it.

#### Node is `Ready` but `kubectl logs` and `osmo workflow logs` hang

**Cause:** the API server cannot reach the node's kubelet on 10250 — the one flow
in this topology that the VPC initiates. Either the node registered with an
address that is not routable from the VPC, or the route to the hybrid networks is
missing.

**Solution:** check that the node's `--node-ip` is its tunnel address and inside
`hybrid_node_cidrs`, and that the private route tables carry both hybrid CIDRs.

#### Small requests succeed but large ones hang

**Symptoms:** `ping` works, TCP connects, an SSH banner appears — and then it
stalls. `ssh -vv` typically stops right after `kex: algorithm: ...`, because
key exchange is the first thing to send a large packet.

**Causes & solutions:**
- **Tunnel MTU too large for the path.** Find the real limit with a
  don't-fragment sweep and set `MTU` on the client and `hybrid_vpn_mtu` on the
  gateway to match the worst path. See `docs/deployment-modes.md`. Note that a
  tunnel MTU left to `wg-quick` on an EC2 instance is derived from a jumbo-frame
  interface and will be far too large.
- **ICMP fragmentation-needed being dropped**, which is what path-MTU discovery
  needs to work at all. Confirm the node security group accepts ICMP from the
  gateway; Terraform adds that rule when it builds the gateway, but a
  user-supplied `hybrid_vpn_gateway_target` needs the equivalent by hand.

## Diagnostic Commands

### Cluster Health
```bash
# Node status
kubectl get nodes -o wide

# All pods in osmo namespace
kubectl get pods -n osmo -o wide

# Events
kubectl get events -n osmo --sort-by='.lastTimestamp'
```

### Terraform State
```bash
cd 001-iac
terraform state list
terraform output
```

### AWS Resources
```bash
# EKS cluster
aws eks describe-cluster --name <cluster-name>

# Node groups
aws eks list-nodegroups --cluster-name <cluster-name>

# RDS
aws rds describe-db-instances --db-instance-identifier <rds-id>

# ElastiCache
aws elasticache describe-replication-groups --replication-group-id <redis-id>
```

### Helm Releases
```bash
helm list -A
helm history osmo-service -n osmo
helm get values osmo-service -n osmo
```

## Getting Help

1. **Check logs:** Always start with pod logs
2. **Review events:** `kubectl get events -n osmo`
3. **Terraform outputs:** Verify infrastructure is correctly deployed
4. **Security groups:** Common cause of connectivity issues
5. **IRSA roles:** Verify service account annotations
