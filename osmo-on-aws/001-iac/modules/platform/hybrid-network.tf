# SPDX-License-Identifier: Apache-2.0

# Routes from the private subnets to the hybrid node and pod networks.
#
# The flow that needs these is VPC-initiated: the API server reaches a hybrid
# node's kubelet on 10250, which is what carries log streaming, exec and
# port-forward. Without a route back out to the tunnel the node registers,
# reports Ready, and every one of those calls hangs.
#
# The pod CIDR is routed as well as the node CIDR so cloud pods and hybrid pods
# can address each other directly.

locals {
  # Gated on the target as well as the flag: without one, aws_route has no target
  # argument to set, and its own error buries the root-module precondition that
  # actually says what to fix.
  hybrid_route_cidrs = var.deploy_hybrid_nodes && var.hybrid_vpn_gateway_target != "" ? concat(var.hybrid_node_cidrs, var.hybrid_pod_cidrs) : []

  # One route per (private route table, hybrid CIDR). Keyed by both so the map
  # stays stable when a route table is added.
  hybrid_routes = {
    for pair in setproduct(range(length(module.vpc.private_route_table_ids)), local.hybrid_route_cidrs) :
    "${pair[0]}-${pair[1]}" => {
      route_table_id = module.vpc.private_route_table_ids[pair[0]]
      cidr_block     = pair[1]
    }
  }

  # aws_route takes the target as a differently named argument per target type.
  hybrid_target_prefix = split("-", var.hybrid_vpn_gateway_target)[0]
}

resource "aws_route" "hybrid" {
  for_each = local.hybrid_routes

  route_table_id         = each.value.route_table_id
  destination_cidr_block = each.value.cidr_block

  gateway_id                = local.hybrid_target_prefix == "vgw" ? var.hybrid_vpn_gateway_target : null
  transit_gateway_id        = local.hybrid_target_prefix == "tgw" ? var.hybrid_vpn_gateway_target : null
  vpc_peering_connection_id = local.hybrid_target_prefix == "pcx" ? var.hybrid_vpn_gateway_target : null
  network_interface_id      = local.hybrid_target_prefix == "eni" ? var.hybrid_vpn_gateway_target : null
}
