#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Deploy hybrid-node support: SSM hybrid activation + Cilium
#
# Opt-in. Does nothing unless 001-iac was applied with should_deploy_hybrid_nodes,
# so scripts 01-05 remain the whole deployment for anyone with no on-premise node.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=defaults.conf
source "$SCRIPT_DIR/defaults.conf"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy the cluster-side half of EKS Hybrid Nodes support.

With no action flag, installs Cilium confined to hybrid nodes. Run that once per
cluster, after the first machine has run 'nodeadm init' — the Cilium operator is
pinned to hybrid nodes and stays Pending until one exists.

OPTIONS:
    -h, --help              Show this help message
    -t, --tf-dir DIR        Terraform directory (default: $DEFAULT_TF_DIR)
    --create-activation     Create an SSM hybrid activation for onboarding a
                            machine, store it in Secrets Manager, and print the
                            secret name. Run once per cluster; re-run to replace
                            an expired one.
    --registration-limit N  Machines the activation may register (default: $DEFAULT_ACTIVATION_REGISTRATION_LIMIT)
    --expiration-hours N    Hours until the activation expires (default: $DEFAULT_ACTIVATION_EXPIRATION_HOURS)
    --tunnel-mtu N          MTU of the link hybrid nodes reach the VPC through.
                            Defaults to the Terraform hybrid_tunnel_mtu output.
    --config-preview        Print configuration and exit

EXAMPLES:
    $(basename "$0") --create-activation
    $(basename "$0")
    $(basename "$0") --tunnel-mtu 1420
EOF
}

# One activation per cluster is what makes the node role's tag-scoped
# ssm:DeregisterManagedInstance permission meaningful, so the limit only needs to
# cover the machines actually being onboarded.
DEFAULT_ACTIVATION_REGISTRATION_LIMIT=2

# The activation is a bootstrap token consumed once, at 'nodeadm init'. After
# that the machine has its own identity and certificate and never uses the code
# again, so a short window is a real reduction in exposure rather than a chore.
DEFAULT_ACTIVATION_EXPIRATION_HOURS=24

# Default values
tf_dir="$SCRIPT_DIR/$DEFAULT_TF_DIR"
create_activation=false
registration_limit="$DEFAULT_ACTIVATION_REGISTRATION_LIMIT"
expiration_hours="$DEFAULT_ACTIVATION_EXPIRATION_HOURS"
tunnel_mtu=""
config_preview=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)              show_help; exit 0 ;;
    -t|--tf-dir)            tf_dir="$2"; shift 2 ;;
    --create-activation)    create_activation=true; shift ;;
    --registration-limit)   registration_limit="$2"; shift 2 ;;
    --expiration-hours)     expiration_hours="$2"; shift 2 ;;
    --tunnel-mtu)           tunnel_mtu="$2"; shift 2 ;;
    --config-preview)       config_preview=true; shift ;;
    *)                      fatal "Unknown option: $1" ;;
  esac
done

# GNU and BSD date disagree on relative-time syntax, and this repo is run from
# both Linux and macOS.
expiration_timestamp() {
  local hours="$1"

  if date -u -d "+${hours} hours" +%Y-%m-%dT%H:%M:%S 2> /dev/null; then
    return 0
  fi
  date -u -v"+${hours}H" +%Y-%m-%dT%H:%M:%S
}

# Stored rather than printed: an activation code on a terminal ends up in scroll
# buffers, shell history and CI logs, and AWS returns it exactly once.
store_activation() {
  local secret_name="$1" region="$2" kms_key_arn="$3" payload="$4"

  if aws secretsmanager describe-secret --secret-id "$secret_name" --region "$region" &> /dev/null; then
    info "Replacing the previous activation in $secret_name"
    aws secretsmanager put-secret-value \
      --secret-id "$secret_name" \
      --secret-string "$payload" \
      --region "$region" > /dev/null
  else
    aws secretsmanager create-secret \
      --name "$secret_name" \
      --description "SSM hybrid activation for EKS hybrid nodes" \
      --secret-string "$payload" \
      --kms-key-id "$kms_key_arn" \
      --region "$region" > /dev/null
  fi
}

do_create_activation() {
  local region="$1" cluster_name="$2" role_name="$3" tag_key="$4" tag_value="$5" kms_key_arn="$6"

  section "Creating SSM Hybrid Activation"

  local expires
  expires=$(expiration_timestamp "$expiration_hours")

  local activation
  activation=$(aws ssm create-activation \
    --region "$region" \
    --default-instance-name "eks-hybrid-nodes" \
    --description "Activation for EKS hybrid nodes" \
    --iam-role "$role_name" \
    --tags "Key=${tag_key},Value=${tag_value}" \
    --registration-limit "$registration_limit" \
    --expiration-date "$expires" \
    --output json)

  local activation_id
  activation_id=$(echo "$activation" | jq -r '.ActivationId')
  [[ -n "$activation_id" && "$activation_id" != "null" ]] || fatal "create-activation returned no ActivationId"

  local secret_name="${cluster_name}/hybrid-nodes/ssm-activation"
  store_activation "$secret_name" "$region" "$kms_key_arn" "$activation"

  print_kv "Activation ID" "$activation_id"
  print_kv "Expires" "$expires (UTC)"
  print_kv "Registration limit" "$registration_limit"
  print_kv "Secret" "$secret_name"
  echo

  info "Read the activation on the machine being onboarded with:"
  info "  aws secretsmanager get-secret-value --region $region \\"
  info "    --secret-id $secret_name --query SecretString --output text"
  info "The code is single-use and expires with the activation. See"
  info "docs/hybrid-nodes-so101.md for the nodeConfig.yaml it goes into."
}

# OSMO asks for the NVIDIA runtime on every task pod, so a platform on a GPU-less
# machine has to name a runtime that node actually has. Cluster-scoped and
# idempotent, so applying it repeatedly is free.
do_apply_runtime_class() {
  info "Ensuring the default OCI runtime is nameable by a pod"
  kubectl apply -f "$SCRIPT_DIR/manifests/runtimeclass-runc.yaml"
}

do_install_cilium() {
  local pod_cidr="$1" mtu="$2"

  section "Installing Cilium for Hybrid Nodes"

  local hybrid_nodes
  hybrid_nodes=$(kubectl get nodes -l "eks.amazonaws.com/compute-type=hybrid" -o name 2> /dev/null | wc -l | tr -d ' ')
  info "Hybrid nodes registered: $hybrid_nodes"

  if [[ "$hybrid_nodes" -eq 0 ]]; then
    warn "No hybrid node has registered yet, so cilium-operator will stay Pending"
    warn "until one does. Run 'nodeadm init' on the machine first."
  fi

  helm_upgrade_install cilium "oci://public.ecr.aws/eks/cilium/cilium" \
    "kube-system" \
    --version "$CILIUM_VERSION" \
    -f "$SCRIPT_DIR/values/cilium-hybrid.yaml" \
    --set "ipam.operator.clusterPoolIPv4PodCIDRList[0]=$pod_cidr" \
    --set "ipam.operator.clusterPoolIPv4MaskSize=$CILIUM_POD_CIDR_MASK_SIZE" \
    --set "MTU=$mtu" \
    --timeout "$HELM_TIMEOUT"

  section "Cilium Status"
  kubectl get pods -n kube-system -l k8s-app=cilium -o wide 2> /dev/null || true
  kubectl get pods -n kube-system -l io.cilium/app=operator -o wide 2> /dev/null || true
  echo
  kubectl get nodes -l "eks.amazonaws.com/compute-type=hybrid" 2> /dev/null || true
}

main() {
  section "Hybrid Nodes Setup"

  require_tools kubectl helm jq aws

  info "Reading Terraform outputs from: $tf_dir"
  local tf_outputs
  tf_outputs=$(read_terraform_outputs "$tf_dir")

  local cluster_name region role_name
  cluster_name=$(tf_require "$tf_outputs" "cluster_name" "EKS cluster name")
  region=$(tf_require "$tf_outputs" "aws_region" "AWS region")
  role_name=$(tf_get "$tf_outputs" "hybrid_nodes_role_name")

  # The role is the marker for the whole feature: it only exists when 001-iac was
  # applied with should_deploy_hybrid_nodes.
  if [[ -z "$role_name" ]]; then
    info "Hybrid nodes are not enabled in 001-iac - nothing to do"
    exit 0
  fi

  local pod_cidr tag_key tag_value kms_key_arn
  pod_cidr=$(echo "$tf_outputs" | jq -r '.hybrid_pod_cidrs.value[0] // empty')
  tag_key=$(echo "$tf_outputs" | jq -r '.hybrid_nodes_activation_tag.value.key // empty')
  tag_value=$(echo "$tf_outputs" | jq -r '.hybrid_nodes_activation_tag.value.value // empty')
  kms_key_arn=$(tf_require "$tf_outputs" "kms_key_arn" "KMS key ARN")

  [[ -n "$pod_cidr" ]] || fatal "hybrid_pod_cidrs is empty in terraform outputs"
  [[ -n "$tag_key" && -n "$tag_value" ]] || fatal "hybrid_nodes_activation_tag is empty in terraform outputs"

  # Never left to Cilium's auto-detection. It reads the MTU of the interface
  # holding the default route, which on a split-tunnel machine is the LAN, not
  # the tunnel — so pod MTU comes out larger than the path can carry and small
  # packets and handshakes keep working while the first large one disappears.
  if [[ -z "$tunnel_mtu" ]]; then
    tunnel_mtu=$(tf_get "$tf_outputs" "hybrid_tunnel_mtu")
  fi
  [[ -n "$tunnel_mtu" ]] || fatal "Tunnel MTU unknown. This deployment routes the hybrid CIDRs at its own gateway, so pass --tunnel-mtu with that link's MTU."

  section "Configuration"
  print_kv "Cluster" "$cluster_name"
  print_kv "Region" "$region"
  print_kv "Hybrid nodes role" "$role_name"
  print_kv "Remote pod CIDR" "$pod_cidr"
  print_kv "Tunnel MTU" "$tunnel_mtu"
  if [[ "$create_activation" == "true" ]]; then
    print_kv "Action" "create SSM hybrid activation"
  else
    print_kv "Action" "install Cilium $CILIUM_VERSION"
  fi

  if [[ "$config_preview" == "true" ]]; then
    info "Config preview mode - exiting"
    exit 0
  fi

  if [[ "$create_activation" == "true" ]]; then
    do_create_activation "$region" "$cluster_name" "$role_name" "$tag_key" "$tag_value" "$kms_key_arn"
    exit 0
  fi

  connect_eks "$region" "$cluster_name"
  do_apply_runtime_class
  do_install_cilium "$pod_cidr" "$tunnel_mtu"

  section "Hybrid Nodes Setup Complete"
  info "Next steps:"
  info "  1. Confirm the hybrid node reports Ready:"
  info "       kubectl get nodes -l eks.amazonaws.com/compute-type=hybrid"
  info "  2. Confirm aws-node did NOT schedule there:"
  info "       kubectl get pods -n kube-system -o wide | grep aws-node"
  info "  3. Add the edge platforms to values/osmo-control-plane.yaml and re-run"
  info "     04-deploy-osmo-control-plane.sh so OSMO can target the machine."
}

main "$@"
