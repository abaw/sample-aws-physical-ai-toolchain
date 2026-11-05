# =============================================================================
# EC2: Headless MuJoCo simulation instance (optional)
# =============================================================================
# One GPU instance on the AWS Deep Learning Base GPU AMI (NVIDIA driver,
# Docker, NVIDIA container toolkit) in a foundation private subnet. Access is
# over SSM Session Manager only: the security group has no inbound rules.
# User data pulls physical-ai/mujoco:latest so it is ready for docker run.
#
# Enable by setting var.enable_sim = true.
# =============================================================================

variable "enable_sim" {
  description = "Create the headless MuJoCo simulation instance"
  type        = bool
  default     = false
}

variable "sim_instance_type" {
  description = "Instance type for the headless simulation instance (GPU required)"
  type        = string
  default     = "g6e.xlarge"
}

variable "sim_volume_size" {
  description = "Root volume size in GB for the headless simulation instance"
  type        = number
  default     = 100
}

variable "sim_ami_ssm_parameter" {
  description = "Public SSM parameter that resolves the Deep Learning Base GPU AMI"
  type        = string
  default     = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
}

data "aws_ssm_parameter" "sim_ami" {
  count = var.enable_sim ? 1 : 0
  name  = var.sim_ami_ssm_parameter
}

data "aws_ssm_parameter" "sim_checkpoints_bucket" {
  count = var.enable_sim ? 1 : 0
  name  = "/${var.project_name}/checkpoints-bucket"
}

data "aws_ssm_parameter" "sim_models_bucket" {
  count = var.enable_sim ? 1 : 0
  name  = "/${var.project_name}/models-bucket"
}

# Security group: egress only (SSM, ECR and S3 are all outbound)
resource "aws_security_group" "sim" {
  count       = var.enable_sim ? 1 : 0
  name        = "${local.prefix}-mujoco-sim-sg"
  description = "MuJoCo headless sim instance (egress only, access via SSM)"
  vpc_id      = local.foundation_vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = {
    Name          = "${local.prefix}-mujoco-sim-sg"
    Project       = var.project_name
    Environment   = var.environment
    Component     = "mujoco-sim"
    "auto-delete" = "no"
  }
}

# IAM role: SSM core + ECR read-only + S3 read on checkpoints and models
resource "aws_iam_role" "sim" {
  count = var.enable_sim ? 1 : 0
  name  = "${local.prefix}-mujoco-sim-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    "auto-delete" = "no"
  }
}

resource "aws_iam_role_policy_attachment" "sim_ssm" {
  count      = var.enable_sim ? 1 : 0
  role       = aws_iam_role.sim[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "sim_ecr" {
  count      = var.enable_sim ? 1 : 0
  role       = aws_iam_role.sim[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy" "sim_s3" {
  count = var.enable_sim ? 1 : 0
  name  = "${local.prefix}-mujoco-sim-s3"
  role  = aws_iam_role.sim[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${data.aws_ssm_parameter.sim_checkpoints_bucket[0].value}",
        "arn:aws:s3:::${data.aws_ssm_parameter.sim_checkpoints_bucket[0].value}/*",
        "arn:aws:s3:::${data.aws_ssm_parameter.sim_models_bucket[0].value}",
        "arn:aws:s3:::${data.aws_ssm_parameter.sim_models_bucket[0].value}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "sim" {
  count = var.enable_sim ? 1 : 0
  name  = "${local.prefix}-mujoco-sim-profile"
  role  = aws_iam_role.sim[0].name

  tags = {
    "auto-delete" = "no"
  }
}

# Runs in the first foundation private subnet; the instance type must be
# offered in that subnet's AZ.
resource "aws_instance" "sim" {
  count                  = var.enable_sim ? 1 : 0
  ami                    = data.aws_ssm_parameter.sim_ami[0].value
  instance_type          = var.sim_instance_type
  subnet_id              = local.foundation_subnet_ids[0]
  vpc_security_group_ids = [aws_security_group.sim[0].id]
  iam_instance_profile   = aws_iam_instance_profile.sim[0].name

  root_block_device {
    volume_size = var.sim_volume_size
    volume_type = "gp3"
    encrypted   = true

    tags = {
      "auto-delete" = "no"
    }
  }

  metadata_options {
    http_tokens = "required" # IMDSv2
  }

  # The ECR URI is <account>.dkr.ecr.<region>.amazonaws.com/<repo>; log in to
  # its registry host, then pull the latest image. User data runs once, so
  # start Docker first and retry while the NAT path comes up.
  user_data = <<-EOF
    #!/bin/bash
    set -euo pipefail
    systemctl enable --now docker
    ECR_URI="${data.aws_ssm_parameter.mujoco_ecr.value}"
    for attempt in 1 2 3 4 5; do
      if aws ecr get-login-password --region ${local.region} \
        | docker login --username AWS --password-stdin "$${ECR_URI%%/*}" \
        && docker pull "$${ECR_URI}:latest"; then
        exit 0
      fi
      sleep $((attempt * 15))
    done
    exit 1
  EOF

  # The AMI parameter moves on every DLAMI release; do not replace the
  # instance each time it does.
  lifecycle {
    ignore_changes = [ami]
  }

  tags = {
    Name          = "${local.prefix}-mujoco-sim"
    Project       = var.project_name
    Environment   = var.environment
    Component     = "mujoco-sim"
    "auto-delete" = "no"
  }
}

output "sim_instance_id" {
  description = "Instance ID of the headless MuJoCo simulation instance"
  value       = var.enable_sim ? aws_instance.sim[0].id : null
}

output "sim_ssm_command" {
  description = "Command to open a shell on the simulation instance over SSM"
  value       = var.enable_sim ? "aws ssm start-session --target ${aws_instance.sim[0].id} --region ${local.region}" : null
}
