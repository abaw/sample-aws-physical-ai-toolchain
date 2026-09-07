# SPDX-License-Identifier: Apache-2.0

# Guardrail for derived resource-name length.
#
# EKS managed node groups name their IAM role "<name_prefix>-<system|gpu>-eks-node-group-",
# and AWS limits an IAM role name_prefix to 38 characters. The longest suffix,
# "-system-eks-node-group-", is 23 chars, so local.name_prefix must be <= 15.
# Fail fast here with an actionable message instead of the cryptic module-level
# "expected length of name_prefix to be in the range (1 - 38)" error.
resource "terraform_data" "name_prefix_length_guard" {
  lifecycle {
    precondition {
      condition     = length(local.name_prefix) <= 15
      error_message = "name_prefix \"${local.name_prefix}\" is ${length(local.name_prefix)} characters, but must be <= 15. It is built as cluster_name-environment-resource_suffix and feeds EKS node-group IAM role names, which AWS caps at 38 characters. Shorten cluster_name and/or resource_suffix (environment is fixed to dev/staging/prod)."
    }
  }
}

# Guardrails for the hybrid node/pod networks.
#
# EKS accepts a remote network configuration that collides with the VPC or the
# service CIDR and only then fails to route, so catch it here. Checked
# unconditionally: the CIDRs are also what a later enable has to live with, and
# the service CIDR is create-only.
resource "terraform_data" "hybrid_network_guard" {
  lifecycle {
    precondition {
      condition     = length(local.hybrid_cidrs_outside_private_space) == 0
      error_message = "hybrid_node_cidrs / hybrid_pod_cidrs must be inside RFC-1918 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) or CGNAT (100.64.0.0/10) space, which AWS requires for remote node and pod networks. Outside that space: ${join(", ", local.hybrid_cidrs_outside_private_space)}."
    }
    precondition {
      condition     = length(local.hybrid_cidrs_overlapping_cluster) == 0
      error_message = "hybrid_node_cidrs / hybrid_pod_cidrs must not overlap vpc_cidr (${var.vpc_cidr}) or cluster_service_ipv4_cidr (${var.cluster_service_ipv4_cidr}): ${join("; ", local.hybrid_cidrs_overlapping_cluster)}."
    }
    precondition {
      condition     = length(local.hybrid_cidrs_overlapping_hybrid) == 0
      error_message = "hybrid_node_cidrs and hybrid_pod_cidrs must not overlap each other: ${join("; ", local.hybrid_cidrs_overlapping_hybrid)}."
    }
    precondition {
      condition     = !local.deploy_hybrid_nodes || var.hybrid_vpn_gateway_target != ""
      error_message = "should_deploy_hybrid_nodes = true needs somewhere to route the hybrid CIDRs. Set hybrid_vpn_gateway_target to an existing vgw-/tgw-/pcx-/eni- id."
    }
  }
}
