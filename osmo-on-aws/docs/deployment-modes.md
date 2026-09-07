# OSMO on AWS - Deployment Modes

OSMO supports three deployment modes to accommodate different architecture requirements.

## Overview

| Mode | Use Case | Components |
|------|----------|------------|
| Full | Single cluster with all components | Control Plane + Compute |
| Control-Plane-Only | Central management cluster | Control Plane only |
| Backend-Only | Remote compute clusters | Compute only |

## Full Mode (Default)

Deploys everything in a single EKS cluster.

### Infrastructure
- VPC with public/private subnets
- EKS cluster with system and GPU node groups
- RDS PostgreSQL
- ElastiCache Redis
- S3 buckets for workflows and datasets

### OSMO Components
- OSMO Service (API)
- OSMO Router
- OSMO Web UI
- OSMO Backend Operator

### Configuration

```hcl
# terraform.tfvars
deployment_mode = "full"
should_deploy_postgresql = true
should_deploy_redis = true
should_deploy_gpu_nodes = true
```

### Deployment

```bash
# Run all scripts
./01-deploy-aws-prerequisites.sh
./02-deploy-gpu-infrastructure.sh
./03-deploy-keycloak.sh --admin-password 'YourSecurePassword!'
./04-deploy-osmo-control-plane.sh --ngc-api-key "your-key"   # applies declarative ConfigMap config
./05-deploy-osmo-backend.sh                                  # creates the osmo-workflow IRSA SA
```

## Control-Plane-Only Mode

For centralized management with remote compute clusters.

### Infrastructure
- VPC with public/private subnets
- EKS cluster with system nodes only (no GPU nodes)
- RDS PostgreSQL
- ElastiCache Redis
- S3 buckets

### OSMO Components
- Consolidated `service` chart: OSMO Service (API) + Router + Web UI + gateway
- No Backend Operator (remote backends connect to this control plane)

### Configuration

```hcl
# terraform.tfvars
deployment_mode = "control-plane-only"
should_deploy_postgresql = true
should_deploy_redis = true
should_deploy_gpu_nodes = false
```

### Deployment

```bash
./01-deploy-aws-prerequisites.sh
# Skip GPU operator
./03-deploy-keycloak.sh --admin-password 'YourSecurePassword!'
./04-deploy-osmo-control-plane.sh --ngc-api-key "your-key"   # applies declarative ConfigMap config
# Skip backend operator
```

### Remote Backend Registration

After control plane deployment, obtain the service URL and use it when deploying backend-only clusters:

```bash
# Get control plane URL
kubectl get ingress -n osmo -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}'
```

## Backend-Only Mode

For distributed compute clusters that connect to a central control plane.

### Infrastructure
- VPC with public/private subnets
- EKS cluster with system and GPU node groups
- S3 buckets (for local workflow data caching)
- No RDS or ElastiCache (uses control plane's databases)

### OSMO Components
- OSMO Backend Operator
- No Control Plane components

### Configuration

```hcl
# terraform.tfvars
deployment_mode = "backend-only"
should_deploy_postgresql = false
should_deploy_redis = false
should_deploy_gpu_nodes = true
external_service_url = "https://osmo-control-plane.example.com"
```

### Deployment

```bash
./01-deploy-aws-prerequisites.sh
./02-deploy-gpu-infrastructure.sh
# Skip control plane
./05-deploy-osmo-backend.sh --service-url "https://osmo-control-plane.example.com"
```

### Backend Token

The backend operator needs a service token from the control plane:

```bash
# On control plane cluster
osmo token create backend-operator --expires-in 365d

# On backend cluster
kubectl create secret generic osmo-operator-token \
  -n osmo \
  --from-literal=token="<TOKEN_FROM_CONTROL_PLANE>"
```

## Hybrid Nodes (on-premise machines in a cloud cluster)

Orthogonal to the three modes above: hybrid nodes add on-premise machines to an
existing cluster instead of giving them a cluster of their own.

Reach for it when a single workflow has to span both — a task on cloud GPUs and a
task on the machine physically wired to a robot, gang-scheduled together. OSMO
maps a pool to exactly one backend cluster and a workflow to one pool, so tasks
in one workflow cannot span two clusters. If the on-premise work is a separate
stage that can run as its own workflow, a backend-only cluster is simpler and
avoids everything below.

### Configuration

```hcl
should_deploy_hybrid_nodes = true
hybrid_node_cidrs          = ["172.16.90.0/29"]   # tunnel addresses nodes register with
hybrid_pod_cidrs           = ["172.16.91.0/24"]   # pod addresses on those nodes
hybrid_vpn_gateway_target  = "vgw-xxxxxxxxxxxxxxxxx"
```

Both CIDRs must be RFC-1918 or CGNAT (`100.64.0.0/10`) and must not overlap
`vpc_cidr` or `cluster_service_ipv4_cidr`; Terraform fails the plan if they do,
because EKS accepts a colliding configuration and only then fails to route.

`cluster_service_ipv4_cidr` is pinned to `172.20.0.0/16` by default. It is
create-only, and once remote networks are configured EKS may choose a
non-default range to avoid a collision — so on an existing cluster set it to the
value `aws eks describe-cluster` already reports. Any other value plans a cluster
replacement.

### What has to be true about the network

Three of the four flows are initiated on-premise and work through any NAT. The
fourth is not: **the API server connects out to each node's kubelet on 10250**,
and that is what carries `kubectl logs`, `exec`, `port-forward`, and OSMO's task
log streaming. Without a route back to the node, it registers, reports `Ready`,
and every one of those calls hangs.

That single flow decides the connectivity option:

| Option | Verdict |
|---|---|
| Direct Connect / Site-to-Site VPN | Correct for a fixed site with a real edge router. Site-to-Site needs a static, internet-routable customer gateway address. |
| AWS Client VPN | Unusable. For IPv4 it source-NATs client traffic to the endpoint's own address, so nothing in the VPC can initiate a connection to the client. |
| Self-managed WireGuard | Works from behind NAT on a dynamic address, which is what a laptop has. `should_deploy_hybrid_vpn_gateway` builds one. |

A node also has to register with its **tunnel** address, not its LAN address.
`nodeadm init` validates this against the declared remote node networks, so the
mistake fails loudly at init rather than turning into a healthy-looking node
whose logs never stream.

### WireGuard gateway

`should_deploy_hybrid_vpn_gateway = true` builds the WireGuard option: one small
arm64 instance in a public subnet with its source/destination check disabled, IP
forwarding on, an elastic IP, and the hybrid routes pointed at its ENI. It is a
separate flag from `should_deploy_hybrid_nodes` so a site with Direct Connect or
a Site-to-Site VPN does not get a second, redundant path — set exactly one of
this flag and `hybrid_vpn_gateway_target`.

```hcl
should_deploy_hybrid_nodes       = true
should_deploy_hybrid_vpn_gateway = true
hybrid_vpn_allowed_cidrs         = ["203.0.113.0/24"]   # where clients dial in from

hybrid_vpn_peers = [
  {
    name        = "lab-machine"
    public_key  = "<the machine's WireGuard public key>"
    allowed_ips = ["172.16.90.2/32", "172.16.91.0/24"]
  },
]
```

Address plan inside `hybrid_node_cidrs`: the gateway takes the **first** address
(`172.16.90.1` for the default `/29`) so the range is on-link on its tunnel
interface, and nodes take addresses from `.2` up. A `/29` leaves six usable
addresses, so assign them deliberately — two machines that pick the same tunnel
address will each work alone and fail together.

Peers are public keys only; each machine keeps its own private key. A peer's
`allowed_ips` is what it is authoritative for: its tunnel address, plus its share
of `hybrid_pod_cidrs` once it is a node running a CNI. A machine that is only
being used to test connectivity needs its `/32` alone.

The gateway generates its own key on first boot and stores it in Parameter Store
as a `SecureString` under the account KMS key, so no private key reaches the
Terraform state, the user data, or this repository. Its public key appears a
minute or two after apply, not during it:

```bash
aws ssm get-parameter \
  --name "$(terraform output -raw hybrid_vpn_server_public_key_parameter)" \
  --query Parameter.Value --output text
```

Editing `hybrid_vpn_peers` and applying is enough — the gateway reconciles from
Parameter Store on a timer, so a peer added later takes effect within about a
minute without replacing the instance. The same timer is why a boot that raced
the elastic IP association recovers on its own. There is no inbound SSH; use
Session Manager, and `journalctl -u hybrid-vpn-sync` for what it did.

### CNI

The AWS VPC CNI cannot serve hybrid nodes: it assigns pods real VPC addresses by
attaching ENIs to the instance, and an on-premise machine has no ENI. AWS's
supported CNI for hybrid nodes is Cilium, taking pod addresses from
`hybrid_pod_cidrs`. Mixed clusters are fine — a CNI is a per-node concern, and
the VPC CNI add-on already carries anti-affinity for the
`eks.amazonaws.com/compute-type: hybrid` label.

### Cost and lifecycle

Hybrid nodes bill per vCPU-hour for as long as they are attached, so detach
between sessions rather than leaving an idle machine joined.

## Multi-Region Architecture

For global deployments:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Control Plane (us-west-2)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ OSMO Service │  │ OSMO Router  │  │  OSMO Web UI │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│           │                                                      │
│  ┌────────┴────────┐  ┌──────────────┐                         │
│  │  RDS PostgreSQL │  │ ElastiCache  │                         │
│  └─────────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Backend (us-east-1)│  │ Backend (eu-west-1)│  │ Backend (ap-northeast-1)│
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │Backend Operator│ │  │ │Backend Operator│ │  │ │Backend Operator│ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ GPU Nodes   │ │  │ │ GPU Nodes   │ │  │ │ GPU Nodes   │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Considerations

### Network Connectivity

- Backend-only clusters need network connectivity to the control plane
- Consider VPC peering, Transit Gateway, or public endpoints
- WebSocket connections (wss://) are used for real-time communication

### Security

- Use private endpoints where possible
- Implement network policies to restrict traffic
- Rotate backend operator tokens regularly
- Use AWS PrivateLink for cross-region connectivity

### Latency

- Place backends close to data sources
- Consider S3 replication for frequently accessed datasets
- Monitor WebSocket connection latency

### Cost Optimization

- Scale GPU nodes to zero when not in use
- Use Spot instances for fault-tolerant workloads
- Share control plane across multiple backend clusters
