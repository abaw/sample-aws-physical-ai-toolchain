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
  # Derived from configuration alone, never from the gateway's attributes: on a
  # first apply the gateway's interface id is unknown, and for_each cannot depend
  # on a value that is only known after apply.
  hybrid_routing_configured = var.deploy_hybrid_nodes && (var.deploy_hybrid_vpn_gateway || var.hybrid_vpn_gateway_target != "")

  hybrid_route_cidrs = local.hybrid_routing_configured ? concat(var.hybrid_node_cidrs, var.hybrid_pod_cidrs) : []

  # One route per (private route table, hybrid CIDR). Keyed by both so the map
  # stays stable when a route table is added.
  hybrid_routes = {
    for pair in setproduct(range(length(module.vpc.private_route_table_ids)), local.hybrid_route_cidrs) :
    "${pair[0]}-${pair[1]}" => {
      route_table_id = module.vpc.private_route_table_ids[pair[0]]
      cidr_block     = pair[1]
    }
  }

  hybrid_route_target = local.hybrid_vpn_enabled ? aws_instance.hybrid_vpn[0].primary_network_interface_id : var.hybrid_vpn_gateway_target

  # aws_route takes the target as a differently named argument per target type.
  hybrid_target_kind = local.hybrid_vpn_enabled ? "eni" : split("-", var.hybrid_vpn_gateway_target)[0]
}

resource "aws_route" "hybrid" {
  for_each = local.hybrid_routes

  route_table_id         = each.value.route_table_id
  destination_cidr_block = each.value.cidr_block

  gateway_id                = local.hybrid_target_kind == "vgw" ? local.hybrid_route_target : null
  transit_gateway_id        = local.hybrid_target_kind == "tgw" ? local.hybrid_route_target : null
  vpc_peering_connection_id = local.hybrid_target_kind == "pcx" ? local.hybrid_route_target : null
  network_interface_id      = local.hybrid_target_kind == "eni" ? local.hybrid_route_target : null
}
