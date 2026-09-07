# SPDX-License-Identifier: Apache-2.0

# WireGuard gateway for hybrid nodes that sit behind NAT on a dynamic address.
#
# The flow that rules out the managed alternatives is VPC-initiated: the API
# server connects out to each node's kubelet. AWS Client VPN source-NATs client
# traffic to the endpoint's own address, so nothing in the VPC can reach the
# client at all; Site-to-Site VPN needs a static, internet-routable customer
# gateway address, which a laptop on a residential line does not have. WireGuard
# holds the tunnel open from the client side with PersistentKeepalive and routes
# in both directions.
#
# A node reached this way also registers with its tunnel address, so it keeps one
# node identity as the machine moves between networks.

locals {
  hybrid_vpn_enabled = var.deploy_hybrid_vpn_gateway && var.deploy_hybrid_nodes

  hybrid_vpn_param_prefix = "/${var.name_prefix}/hybrid-vpn"
  hybrid_vpn_key_param    = "${local.hybrid_vpn_param_prefix}/server-private-key"
  hybrid_vpn_pub_param    = "${local.hybrid_vpn_param_prefix}/server-public-key"
  hybrid_vpn_peers_param  = "${local.hybrid_vpn_param_prefix}/peers"

  # The gateway takes the first address of the remote node network, so the whole
  # range is on-link on its tunnel interface and node addresses start at .2.
  hybrid_vpn_node_cidr = length(var.hybrid_node_cidrs) > 0 ? var.hybrid_node_cidrs[0] : ""
  hybrid_vpn_tunnel_address = local.hybrid_vpn_node_cidr == "" ? "" : format(
    "%s/%s", cidrhost(local.hybrid_vpn_node_cidr, 1), split("/", local.hybrid_vpn_node_cidr)[1]
  )

  hybrid_vpn_peer_config = length(var.hybrid_vpn_peers) > 0 ? join("\n", [
    for p in var.hybrid_vpn_peers : <<-EOT
    [Peer]
    # ${p.name}
    PublicKey = ${p.public_key}
    AllowedIPs = ${join(",", p.allowed_ips)}
    EOT
  ]) : "# no peers registered"
}

# Canonical publishes the current image id per region, so this stays
# region-agnostic. Pinned after creation deliberately: replacing the gateway
# during an unrelated apply would drop every tunnel through it, so an image
# refresh is a deliberate act (taint the instance). The server key survives it —
# it lives in Parameter Store, not on the disk.
data "aws_ssm_parameter" "hybrid_vpn_ami" {
  count = local.hybrid_vpn_enabled && var.hybrid_vpn_ami_id == "" ? 1 : 0

  name = "/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id"
}

resource "aws_security_group" "hybrid_vpn" {
  count = local.hybrid_vpn_enabled ? 1 : 0

  name        = "${var.name_prefix}-hybrid-vpn-sg"
  description = "WireGuard gateway for hybrid nodes"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = var.hybrid_vpn_listen_port
    to_port     = var.hybrid_vpn_listen_port
    protocol    = "udp"
    cidr_blocks = var.hybrid_vpn_allowed_cidrs
    description = "WireGuard from the networks hybrid nodes dial in from"
  }

  # Traffic the gateway forwards is evaluated as inbound on its own interface, so
  # without this the VPC-initiated half of every flow is dropped here rather than
  # at the node.
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [var.vpc_cidr]
    description = "VPC traffic destined for the hybrid networks"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-hybrid-vpn-sg"
  })
}

resource "aws_iam_role" "hybrid_vpn" {
  count = local.hybrid_vpn_enabled ? 1 : 0

  name = "${var.name_prefix}-hybrid-vpn"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = var.common_tags
}

# Session Manager, so the gateway needs no inbound SSH and no key pair.
resource "aws_iam_role_policy_attachment" "hybrid_vpn_ssm_core" {
  count = local.hybrid_vpn_enabled ? 1 : 0

  role       = aws_iam_role.hybrid_vpn[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "hybrid_vpn" {
  count = local.hybrid_vpn_enabled ? 1 : 0

  name = "${var.name_prefix}-hybrid-vpn"
  role = aws_iam_role.hybrid_vpn[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "GatewayParameters"
        Effect = "Allow"
        Action = ["ssm:GetParameter", "ssm:PutParameter"]
        Resource = [
          "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${local.hybrid_vpn_param_prefix}/*"
        ]
      },
      {
        Sid      = "ServerKeyEncryption"
        Effect   = "Allow"
        Action   = ["kms:Encrypt", "kms:Decrypt", "kms:DescribeKey"]
        Resource = [aws_kms_key.osmo.arn]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "hybrid_vpn" {
  count = local.hybrid_vpn_enabled ? 1 : 0

  name = "${var.name_prefix}-hybrid-vpn"
  role = aws_iam_role.hybrid_vpn[0].name

  tags = var.common_tags
}

# Public keys only — the peers' private keys stay on their own machines, and this
# is a plain String parameter so it is readable without decrypt permission.
resource "aws_ssm_parameter" "hybrid_vpn_peers" {
  count = local.hybrid_vpn_enabled ? 1 : 0

  name        = local.hybrid_vpn_peers_param
  description = "WireGuard peer blocks for the hybrid-node gateway"
  type        = "String"
  value       = local.hybrid_vpn_peer_config

  tags = var.common_tags
}

resource "aws_instance" "hybrid_vpn" {
  count = local.hybrid_vpn_enabled ? 1 : 0

  ami                         = var.hybrid_vpn_ami_id != "" ? var.hybrid_vpn_ami_id : data.aws_ssm_parameter.hybrid_vpn_ami[0].value
  instance_type               = var.hybrid_vpn_instance_type
  subnet_id                   = module.vpc.public_subnets[0]
  vpc_security_group_ids      = [aws_security_group.hybrid_vpn[0].id]
  iam_instance_profile        = aws_iam_instance_profile.hybrid_vpn[0].name
  associate_public_ip_address = true

  # Mandatory: the gateway sends and receives packets addressed to the hybrid
  # networks, which EC2 drops by default as spoofed.
  source_dest_check = false

  # Everything this instance is runs out of its user data, and the provider
  # otherwise applies a user-data change to the attribute only, leaving the
  # running host on the old script until someone stops and starts it. Replacing
  # is cheap: the server key is in Parameter Store, not on the disk.
  user_data_replace_on_change = true

  user_data = templatefile("${path.module}/templates/hybrid-vpn-userdata.sh.tftpl", {
    aws_region     = var.aws_region
    kms_key_id     = aws_kms_key.osmo.arn
    key_param      = local.hybrid_vpn_key_param
    pub_param      = local.hybrid_vpn_pub_param
    peers_param    = local.hybrid_vpn_peers_param
    tunnel_address = local.hybrid_vpn_tunnel_address
    listen_port    = var.hybrid_vpn_listen_port
    mtu            = var.hybrid_vpn_mtu
    sync_script    = file("${path.module}/files/hybrid-vpn-sync.sh")
  })

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_size = 8
    volume_type = "gp3"
    encrypted   = true
    kms_key_id  = aws_kms_key.osmo.arn
  }

  lifecycle {
    ignore_changes = [ami]
  }

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-hybrid-vpn"
  })
}

# A fixed address because it is what every client configuration names as its
# endpoint; an auto-assigned public address changes on stop/start.
resource "aws_eip" "hybrid_vpn" {
  count = local.hybrid_vpn_enabled ? 1 : 0

  domain = "vpc"

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-hybrid-vpn"
  })
}

resource "aws_eip_association" "hybrid_vpn" {
  count = local.hybrid_vpn_enabled ? 1 : 0

  instance_id   = aws_instance.hybrid_vpn[0].id
  allocation_id = aws_eip.hybrid_vpn[0].id
}
