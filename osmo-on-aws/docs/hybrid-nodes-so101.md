# Onboarding an on-premise machine as a hybrid node

One-time infrastructure setup: turn a machine on your own network into a node of
the EKS cluster, so a workflow can gang-schedule a task on it alongside tasks on
cloud GPUs. The reference case is a laptop with a robot arm on USB, but nothing
here is specific to that.

This is the per-machine procedure. The cluster-side prerequisites — remote network
configuration, VPC routes, the tunnel gateway, the node IAM role — are covered in
[deployment-modes.md](deployment-modes.md#hybrid-nodes-on-premise-machines-in-a-cloud-cluster)
and must be in place first.

Expect about 30 minutes for a machine you have not onboarded before.

## 1. Prerequisites

**Operating system.** AWS validates hybrid nodes on Ubuntu 20.04/22.04/24.04, RHEL
8/9, AL2023, and Bottlerocket (vSphere only). AWS states that anything outside that
list "is not covered by AWS Support".

> The reference machine for this repo's SO-101 example runs **Ubuntu 26.04, which
> is not on that list.** That is a deliberate choice for a single lab machine, not
> a recommendation. Everything below has been exercised on it, but if you want AWS
> support for the node, use a validated release.

Cilium additionally needs Linux ≥ 5.10, which rules out Ubuntu 20.04 and RHEL 8
even though AWS validates them for hybrid nodes generally.

**Network.** AWS guidance is ≥ 100 Mbps and ≤ 200 ms round-trip to the Region.
Nothing about a robot control loop is comfortable near that latency ceiling — for
that workload treat 200 ms as a hard failure point, not a target.

**Swap.** `nodeadm init` disables **file-based** swap itself and says so. A swap
**partition** it will not touch, so run `swapoff -a` and remove the entry from
`/etc/fstab` first. Do not pre-emptively disable file swap; there is no need.

**Architecture.** On arm64, kube-proxy requires ARMv8.2 with crypto extensions.

## 2. Bring up the tunnel

The machine needs an address inside the cluster's `hybrid_node_cidrs` and a route
to the VPC. With the WireGuard gateway this repo can build, the client config is:

```ini
[Interface]
Address = 10.90.90.2/32      # inside hybrid_node_cidrs; .1 is the gateway
MTU = 1280                   # see below — not optional
PrivateKey = <this machine's private key>

[Peer]
PublicKey = <gateway public key, from the SSM parameter>
Endpoint = <gateway elastic IP>:51820
AllowedIPs = 10.0.0.0/16, 10.90.90.0/29, 172.20.0.0/16
PersistentKeepalive = 25
```

Register this machine's **public** key in `hybrid_vpn_peers` and apply; the gateway
reconciles from Parameter Store within about a minute.

`PersistentKeepalive` is what holds the tunnel open from behind NAT, which is the
only reason the VPC can initiate connections inward at all.

**What `AllowedIPs` covers, and why each entry is there:**

- **The VPC CIDR** — the API server, and pods on cloud nodes.
- **The node CIDR** — other machines on the same tunnel.
- **The service CIDR** — needed, and not obvious. kube-proxy rewrites a ClusterIP
  to a VPC address, but the kernel's route lookup happens *before* that rewrite. If
  the service CIDR has no route through the tunnel, the lookup lands on the LAN
  interface and picks its address as the source; the rewritten packet then leaves
  through the tunnel carrying a source address the VPC has no route back to.
  Everything that talks to a Service breaks, while direct connections to VPC
  addresses keep working. Nothing with a service address ever crosses the wire —
  the entry exists purely to create the route.
- **Not `0.0.0.0/0`** — the machine keeps its own internet path, and nothing is
  billed through the gateway.
- **Not the pod CIDR** — this machine owns that range locally once its CNI is
  running. The gateway's peer entry is the mirror image and *does* carry it.

**MTU is not optional.** Left to `wg-quick`, it is derived from the interface
holding the default route and will be wrong. Measure the path rather than trusting
a default: sweep a don't-fragment ping and add 60 to the largest inner size that
gets through to get the wire size. Measured limits vary a lot by site — one
corporate path capped the wire at 1388 where an office path carried 1480.

The failure mode reads as a healthy network: ping works, TCP connects, protocol
banners arrive, and then the first large packet vanishes. `ssh -vv` characteristically
stalls right after `kex: algorithm: …`, key exchange being the first thing to send
over a kilobyte.

## 3. Verify connectivity both ways before going further

Cheapest possible gate, and it localises a whole class of later confusion.

```bash
ping <gateway tunnel address>                  # from the machine
curl -sk https://<a cluster ENI>:443/          # kubelet -> API server path
```

EKS opens **only TCP 443** to remote networks on the cluster security group, so a
cluster ENI never answers `ping`. Use 443; a timeout there is not evidence of a
broken tunnel.

Then the direction that matters most — a pod on a cloud node reaching this
machine's tunnel address, which is the flow that carries log streaming and `exec`:

```bash
kubectl run probe --image=public.ecr.aws/docker/library/busybox:1.37 \
  --restart=Never -- ping -c3 10.90.90.2
```

If nothing on the machine answers, distinguish two cases before suspecting the
infrastructure: a **closed** port that returns RST means packets arrive and only a
listener is missing, whereas a closed port that **times out** means something drops
them before the IP stack — usually a host firewall on the machine itself.

## 4. Install `nodeadm`

```bash
curl -OL 'https://hybrid-assets.eks.amazonaws.com/releases/latest/bin/linux/amd64/nodeadm'
chmod +x nodeadm
sudo ./nodeadm install 1.35 --credential-provider ssm --region <cluster region>
```

Use `latest`: SSM requires `nodeadm` ≥ 1.0.19, and earlier builds carry an outdated
SSM signing key and fail with `Signature Verification Error: No matching signature`.
The Kubernetes version must match the cluster's.

**Pass `--region` explicitly.** It defaults to `us-west-2` for regional artifacts
regardless of where the cluster is, so the SSM installer is otherwise fetched
cross-region.

`--containerd-source` defaults to `distro`, which is right on current Ubuntu.
Two caveats: it is unsupported on RHEL (use `docker` there), and `docker` is
unsupported on AL2023. On Ubuntu, containerd must be ≥ 1.7.19 — earlier 1.7.x has
an AppArmor defect that leaves pods stuck `Terminating` rather than failing at
install (Ubuntu bug #2065423). Anything shipping containerd 2.x is clear of it.

## 5. Write `nodeConfig.yaml`

Create the activation first, from a machine with AWS credentials — not this one:

```bash
002-setup/06-deploy-hybrid-nodes.sh --create-activation
```

It stores the code and id in Secrets Manager and prints only the secret name. The
code is single-use, expires with the activation, and AWS returns it exactly once,
so never paste it into a terminal, a file under version control, or this document.

```yaml
apiVersion: node.eks.aws/v1alpha1
kind: NodeConfig
spec:
  cluster:
    name: <cluster>
    region: <region>
  kubelet:
    flags:
      - --node-ip=10.90.90.2                                # the tunnel address
      - --node-labels=osmo/node-type=so101-edge
      - --register-with-taints=osmo/edge=so101:NoSchedule
  hybrid:
    ssm:
      activationCode: <code>
      activationId: <id>
```

`--node-ip` is the flag most worth getting right. A split-tunnel setup leaves the
default route on the LAN, so kubelet's own choice of node IP is the LAN address,
which is meaningless inside the VPC — and the symptom is not lost connectivity but
a node that registers, reports `Ready`, and then hangs every `kubectl logs` and
`exec`. `nodeadm init` validates the node interface, so a wrong address fails
loudly at init; treat that failure as a real problem rather than reaching for
`--skip`.

The label and taint belong here rather than in a later `kubectl` call. A taint
applied after the node is already `Ready` leaves a window in which arbitrary
cluster workloads can schedule onto a machine wired to a robot.

Check, then initialise:

```bash
sudo ./nodeadm config check -c file://nodeConfig.yaml
sudo ./nodeadm init -c file://nodeConfig.yaml
```

`nodeadm debug` is a **post-init** tool. Run before the machine is registered it
has no credentials to use, falls back to IMDS, and fails in a way that says nothing
about your configuration.

The phases `--skip` accepts: `install-validation, cni-validation,
node-ip-validation, credentials-validation, kubelet-cert-validation,
ssm-api-network-validation, iam-ra-api-network-validation, aws-auth-validation,
k8s-endpoint-network-validation, k8s-authentication-validation,
kubelet-version-skew-validation, api-server-endpoint-resolution-validation,
proxy-validation, node-inactive-validation, preprocess, config, run`.

## 6. Expect `NotReady`

The node joins without a CNI and stays `NotReady` until one runs. That is normal
and is the next step.

## 7. Note the node name

With SSM the node name is **not configurable** — it is generated as
`mi-0123456789abcdef0`. Nothing may key off a hostname: not `kubectl` invocations,
not a `nodeSelector`, not a pod template. Select on the label from step 5.

## 8. Install the CNI

Once at least one hybrid node has registered:

```bash
002-setup/06-deploy-hybrid-nodes.sh
```

This installs Cilium confined to hybrid nodes, with the pod CIDR and MTU taken from
Terraform outputs. Run it *after* `nodeadm init`, not before: the Cilium operator is
pinned to hybrid nodes and has nowhere to run until one exists.

The node should reach `Ready` within a minute or two. Confirm the CNIs stayed
separate — no `aws-node` pod should appear on the hybrid node:

```bash
kubectl get nodes -l eks.amazonaws.com/compute-type=hybrid
kubectl get pods -n kube-system -o wide | grep -E 'aws-node|cilium'
```

A pod on a hybrid node reaches other pods by masquerading to the **node** address,
so anything checking source addresses on the cloud side sees the node CIDR, not the
pod CIDR.

## 9. Device plumbing

Serial and camera devices are numbered in the order the kernel finds them, so
`/dev/ttyACM0` versus `ACM1` depends on plug order. Identify each device by
something stable before writing it into a workflow.

**Serial devices** carry a USB serial number, so `/dev/serial/by-id/` distinguishes
them:

```console
$ ls -l /dev/serial/by-id/
usb-1a86_USB_Single_Serial_XXXXXXXXXX-if00 -> ../../ttyACM0
usb-1a86_USB_Single_Serial_YYYYYYYYYY-if00 -> ../../ttyACM1
```

For a leader/follower pair, both are the same adapter with the same six servos, so
the serial number alone does not say which arm is which. **Supply voltage does**, and
reading it needs no motion: a follower runs from a 12 V supply, a leader is bus-powered
at 5 V. Reading each servo's voltage register is a read-only bus operation.

**Cameras are harder, because identical models share an identical `by-id` string** —
the descriptor carries no serial number, so the two symlinks collide and only one
appears. Use `/dev/v4l/by-path/` instead, which is derived from the USB port and
therefore stable as long as nothing is re-plugged.

Each camera also claims **two** `/dev/video*` nodes, and only one carries frames:

```console
$ v4l2-ctl -d /dev/videoN --all | grep -A2 'Device Caps'
                Video Capture      # this one
                Metadata Capture   # not this one
```

Which of a pair is which is not fixed either, so check rather than assume the lower
number. Finally, tell the two cameras apart by **looking at a frame** — capture one
from each and see which shows the workspace and which looks out past the gripper:

```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 -i /dev/videoN -frames:v 1 out.jpg
```

Record the resolved paths: they become the platform's `allowed_mounts` and the
workflow's `volumeMounts`.

**A task must run privileged to use any of them.** The platform grants
`privileged_allowed`, and the task has to ask for `privileged: true`. Without it the
device appears in the container with correct ownership and every open fails, because
the container device cgroup blocks it — which reads as a broken device rather than a
missing permission.

## 10. What consumes the node once it is attached

Two workflows in this repo schedule a task here, both as the robot half of a
gang-scheduled pair whose other half holds a cloud GPU:
[`so101-vla/04_hil_eval.yaml`](../003-workflows/so101-vla/README.md) and
[`so101-pi0fast/04_hil_eval.yaml`](../003-workflows/so101-pi0fast/README.md). They share one
robot client and differ only in the policy server they talk to, so the device paths
identified in step 9 are set the same way in both — as workflow variables, not baked
into a script.

Both refuse to move the arm unless told to explicitly. Rehearse with `no_motion=true`
first; it exercises scheduling, the tunnel, device access and the full control loop
while performing zero writes to the servo bus.

## 11. Detach and re-attach

Hybrid nodes bill **per vCPU-hour for as long as they are attached**, so remove the
node between sessions rather than leaving an idle machine joined.

```bash
kubectl drain <mi-…> --ignore-daemonsets --delete-emptydir-data
kubectl delete node <mi-…>
sudo ./nodeadm uninstall            # also deregisters the SSM managed instance
```

Re-onboarding needs a fresh activation if the old one has expired, then steps 5
onward again.

## 12. Troubleshooting

**`nodeadm init` fails node-interface validation.** `--node-ip` is outside the
cluster's declared `hybrid_node_cidrs`, or is not an address on this machine.

**Node stays `NotReady`.** Expected before step 8. After it, check the Cilium agent
on that node.

**Node is `Ready` but `kubectl logs` and `exec` hang.** The API server cannot reach
kubelet on 10250 — the one flow the VPC initiates. Check that `--node-ip` is the
tunnel address, that the private route tables carry both hybrid CIDRs, and that the
route target is reachable.

**Cilium will not start: `Unable to contact k8s api-server … 172.20.0.1 … i/o
timeout`.** The service CIDR is missing from the tunnel's `AllowedIPs` — see step 2.
Confirm with `ip route get <a ClusterIP>`: if it resolves via the LAN interface
rather than the tunnel, that is the cause. Binding a test client to the tunnel
address succeeds where an unbound one times out, which is the quickest way to tell
this apart from a genuine outage.

**DNS fails from a pod on the hybrid node, or a pod cannot reach a Service.** The
CoreDNS replicas run on cloud nodes, so that traffic crosses the node security
group. Enabling hybrid nodes opens it for the hybrid CIDRs; a deployment supplying
its own route target needs the equivalent.

**Pods stuck `Terminating`.** On Ubuntu, the containerd/AppArmor defect in step 4.
This looks like a tunnel or CNI problem and is neither.

**Large transfers hang while small requests succeed.** MTU. See step 2, and confirm
ICMP fragmentation-needed is not being dropped by a firewall on this machine — that
is what path-MTU discovery depends on.

**Image pulls fail with an authorization error.** The node IAM role is missing ECR
pull permission.

**Serial device busy.** A stale process on the host still holds it.

**A workflow task on the node never starts, or hangs after its script finishes.**
See `docs/troubleshooting.md` — the two causes are the NVIDIA runtime class OSMO
puts on every task pod, and pods being unable to reach the STS endpoint that IRSA
depends on. Neither is visible from the node itself.

For any of these once the node is registered, start with
`sudo ./nodeadm debug -c file://nodeConfig.yaml`: it checks AWS API reachability,
that the role's credentials can be obtained, that the cluster endpoint and its
certificate are reachable, and that the node's cluster identity is valid.
