# SPDX-License-Identifier: Apache-2.0

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.name_prefix}-vpc"
  cidr = var.vpc_cidr

  azs                 = var.azs
  private_subnets     = var.private_subnets
  public_subnets      = var.public_subnets
  database_subnets    = var.database_subnets
  elasticache_subnets = var.elasticache_subnets

  # NAT Gateway configuration
  enable_nat_gateway     = true
  single_nat_gateway     = var.single_nat_gateway
  one_nat_gateway_per_az = !var.single_nat_gateway
  enable_vpn_gateway     = false

  # DNS configuration
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Subnet groups for managed services
  create_database_subnet_group    = true
  database_subnet_group_name      = "${var.name_prefix}-db-subnet-group"
  create_elasticache_subnet_group = true
  elasticache_subnet_group_name   = "${var.name_prefix}-elasticache-subnet-group"

  # Tags for EKS and ALB Controller discovery
  public_subnet_tags = {
    "kubernetes.io/role/elb"                    = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"           = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }

  tags = var.common_tags
}

# VPC Endpoints for AWS services (cost optimization and security)
module "vpc_endpoints" {
  source  = "terraform-aws-modules/vpc/aws//modules/vpc-endpoints"
  version = "~> 5.0"

  vpc_id = module.vpc.vpc_id

  endpoints = {
    s3 = {
      service         = "s3"
      service_type    = "Gateway"
      route_table_ids = module.vpc.private_route_table_ids
      tags            = { Name = "${var.name_prefix}-s3-endpoint" }
    }
    ecr_api = {
      service             = "ecr.api"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [aws_security_group.vpc_endpoints.id]
      tags                = { Name = "${var.name_prefix}-ecr-api-endpoint" }
    }
    ecr_dkr = {
      service             = "ecr.dkr"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [aws_security_group.vpc_endpoints.id]
      tags                = { Name = "${var.name_prefix}-ecr-dkr-endpoint" }
    }
    sts = {
      service             = "sts"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [aws_security_group.vpc_endpoints.id]
      tags                = { Name = "${var.name_prefix}-sts-endpoint" }
    }
    secretsmanager = {
      service             = "secretsmanager"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [aws_security_group.vpc_endpoints.id]
      tags                = { Name = "${var.name_prefix}-secretsmanager-endpoint" }
    }
    # CloudWatch Logs — keeps Fluent Bit log shipping private (off NAT).
    logs = {
      service             = "logs"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [aws_security_group.vpc_endpoints.id]
      tags                = { Name = "${var.name_prefix}-logs-endpoint" }
    }
    # Amazon Managed Prometheus (remote-write target for the managed scraper).
    aps_workspaces = {
      service             = "aps-workspaces"
      private_dns_enabled = true
      subnet_ids          = module.vpc.private_subnets
      security_group_ids  = [aws_security_group.vpc_endpoints.id]
      tags                = { Name = "${var.name_prefix}-aps-workspaces-endpoint" }
    }
  }

  tags = var.common_tags
}

# Security group for VPC endpoints
resource "aws_security_group" "vpc_endpoints" {
  name        = "${var.name_prefix}-vpc-endpoints-sg"
  description = "Security group for VPC endpoints"
  vpc_id      = module.vpc.vpc_id

  # The hybrid CIDRs are here because these endpoints have private DNS enabled,
  # which makes them the only way a pod on a hybrid node can reach the services
  # behind them. Such a pod resolves through CoreDNS and therefore the VPC
  # resolver, so it gets the endpoint's private addresses and routes to them
  # through the tunnel — where this group is what accepts or drops it.
  #
  # IRSA is what makes this load-bearing: a pod exchanges its projected token at
  # STS, and with no route to STS the exchange hangs rather than failing, so a
  # task looks like it is working and never finishes. Note the node itself is
  # unaffected and so hides the problem: it resolves through its own network's
  # resolver, gets public addresses, and pulls images perfectly well.
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = concat([var.vpc_cidr], var.deploy_hybrid_nodes ? concat(var.hybrid_node_cidrs, var.hybrid_pod_cidrs) : [])
    description = "HTTPS from the VPC and any hybrid networks"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-vpc-endpoints-sg"
  })
}
