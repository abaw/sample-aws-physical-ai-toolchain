# SPDX-License-Identifier: Apache-2.0

# Node-level credentials for hybrid nodes.
#
# An EC2 node gets credentials from its instance profile via IMDS. A machine
# outside AWS has no IMDS, so kubelet and nodeadm assume this role instead —
# through an SSM hybrid activation, which is what a machine presents in place of
# an instance identity.
#
# Only the role is here. The activation itself is a single-use bootstrap token
# that AWS returns once and that expires in 24 hours, so it is created by
# 002-setup/06-deploy-hybrid-nodes.sh: as a Terraform resource it would park a
# credential in state for the life of the cluster and offer a diff on every plan.
#
# Pod-level credentials need nothing here. IRSA exchanges a projected service
# account token at STS and never touches IMDS, so workflow pods on a hybrid node
# keep using the same service account they use on a cloud node.

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

locals {
  hybrid_nodes_enabled = var.deploy_hybrid_nodes

  # The tag that scopes ssm:DeregisterManagedInstance below. The activation
  # carries the same tag, which is what limits nodeadm uninstall to instances
  # belonging to this cluster.
  hybrid_nodes_activation_tag_key   = "EKSClusterARN"
  hybrid_nodes_activation_tag_value = module.eks.cluster_arn
}

resource "aws_iam_role" "hybrid_nodes" {
  count = local.hybrid_nodes_enabled ? 1 : 0

  name        = "${var.name_prefix}-hybrid-nodes"
  description = "Node-level credentials for EKS hybrid nodes, assumed through an SSM hybrid activation"

  # Both conditions are load-bearing. A role that trusts ssm.amazonaws.com
  # unconditionally can be assumed on behalf of any account's SSM, so an
  # activation created elsewhere would yield credentials for this cluster.
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ssm.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = {
        StringEquals = {
          "aws:SourceAccount" = data.aws_caller_identity.current.account_id
        }
        ArnEquals = {
          "aws:SourceArn" = "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
        }
      }
    }]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy" "hybrid_nodes" {
  count = local.hybrid_nodes_enabled ? 1 : 0

  name = "${var.name_prefix}-hybrid-nodes"
  role = aws_iam_role.hybrid_nodes[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Without DescribeCluster the node configuration has to carry the API
        # endpoint, the CA bundle and the service CIDR by hand; without
        # ListAccessEntries, nodeadm init needs --skip cluster-access-validation
        # and so joins without checking it is allowed to.
        Sid      = "ClusterDiscovery"
        Effect   = "Allow"
        Action   = ["eks:DescribeCluster", "eks:ListAccessEntries"]
        Resource = module.eks.cluster_arn
      },
      {
        Sid      = "DescribeOwnRegistration"
        Effect   = "Allow"
        Action   = ["ssm:DescribeInstanceInformation"]
        Resource = "*"
      },
      {
        # Tag-scoped so nodeadm uninstall can retire this cluster's machines and
        # no other managed instance in the account.
        Sid      = "DeregisterOwnRegistration"
        Effect   = "Allow"
        Action   = ["ssm:DeregisterManagedInstance"]
        Resource = "arn:aws:ssm:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:managed-instance/*"
        Condition = {
          StringEquals = {
            "ssm:resourceTag/${local.hybrid_nodes_activation_tag_key}" = local.hybrid_nodes_activation_tag_value
          }
        }
      }
    ]
  })
}

# Registers the machine with SSM and keeps its hour-long credentials rotating.
resource "aws_iam_role_policy_attachment" "hybrid_nodes_ssm_core" {
  count = local.hybrid_nodes_enabled ? 1 : 0

  role       = aws_iam_role.hybrid_nodes[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Image pulls. Missing this surfaces as an authorization failure from the
# registry at pull time, which reads as a registry problem rather than a node
# permission one.
resource "aws_iam_role_policy_attachment" "hybrid_nodes_ecr_pull" {
  count = local.hybrid_nodes_enabled ? 1 : 0

  role       = aws_iam_role.hybrid_nodes[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPullOnly"
}

# Maps the role to node RBAC. HYBRID_LINUX grants the same Kubernetes identity a
# managed node group's instance role gets, so no access policy is associated —
# EKS derives the permissions from the type.
#
# Declared here rather than in the cluster module's access_entries map: the role
# it points at is scoped by the cluster ARN, so routing the entry back through
# the module would make the module and the role depend on each other.
resource "aws_eks_access_entry" "hybrid_nodes" {
  count = local.hybrid_nodes_enabled ? 1 : 0

  cluster_name  = module.eks.cluster_name
  principal_arn = aws_iam_role.hybrid_nodes[0].arn
  type          = "HYBRID_LINUX"

  tags = var.common_tags
}
