# =============================================================================
# AWS BATCH: Single-Node MuJoCo RL Training (optional)
# =============================================================================
# Provisions a Batch compute environment with GPU instances that scales to
# zero when idle, a job queue, and a single-node job definition that runs
# Brax PPO on a MuJoCo Playground env (containers/mujoco/batch-train).
#
# Enable by setting var.enable_batch = true. Runs in the foundation VPC.
# =============================================================================

variable "enable_batch" {
  description = "Enable AWS Batch compute environment for MuJoCo RL training"
  type        = bool
  default     = false
}

variable "batch_instance_type" {
  description = "Instance type for Batch compute (1-GPU G6e, L40S)"
  type        = string
  default     = "g6e.4xlarge"
}

variable "batch_max_vcpus" {
  description = "Maximum vCPUs for the Batch compute environment"
  type        = number
  default     = 32
}

variable "batch_job_vcpus" {
  description = "vCPUs reserved per training job"
  type        = number
  default     = 8
}

variable "batch_job_memory_mib" {
  description = "Memory (MiB) reserved per training job"
  type        = number
  default     = 32768
}

variable "batch_job_timeout_seconds" {
  description = "Per-attempt timeout, so a hung job does not keep a GPU instance billing"
  type        = number
  default     = 14400
}

# Foundation buckets: checkpoints (14-day expiry) and exported models
data "aws_ssm_parameter" "checkpoints_bucket" {
  count = var.enable_batch ? 1 : 0
  name  = "/${var.project_name}/checkpoints-bucket"
}

data "aws_ssm_parameter" "models_bucket" {
  count = var.enable_batch ? 1 : 0
  name  = "/${var.project_name}/models-bucket"
}

locals {
  batch_checkpoints_bucket = var.enable_batch ? data.aws_ssm_parameter.checkpoints_bucket[0].value : ""
  batch_models_bucket      = var.enable_batch ? data.aws_ssm_parameter.models_bucket[0].value : ""
}

# Security group for Batch compute nodes: single-node jobs, so no ingress
resource "aws_security_group" "batch" {
  count       = var.enable_batch ? 1 : 0
  name        = "${local.prefix}-mujoco-batch-sg"
  description = "MuJoCo Batch RL compute: egress only"
  vpc_id      = local.foundation_vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = {
    Name        = "${local.prefix}-mujoco-batch-sg"
    Project     = var.project_name
    Environment = var.environment
    Component   = "mujoco"
  }
}

# IAM role for Batch compute instances (ECS agent, image pull, logs)
resource "aws_iam_role" "batch_instance" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-mujoco-batch-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_ecs" {
  count      = var.enable_batch ? 1 : 0
  role       = aws_iam_role.batch_instance[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-mujoco-batch-profile"
  role  = aws_iam_role.batch_instance[0].name
}

# IAM role for the training container: S3 on the two foundation buckets only
resource "aws_iam_role" "batch_job" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-mujoco-batch-job-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "batch_job_s3" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-mujoco-batch-job-s3"
  role  = aws_iam_role.batch_job[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${local.batch_checkpoints_bucket}",
          "arn:aws:s3:::${local.batch_models_bucket}"
        ]
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"]
        Resource = [
          "arn:aws:s3:::${local.batch_checkpoints_bucket}/*",
          "arn:aws:s3:::${local.batch_models_bucket}/*"
        ]
      }
    ]
  })
}

# Batch service role
resource "aws_iam_role" "batch_service" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-mujoco-batch-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "batch.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  count      = var.enable_batch ? 1 : 0
  role       = aws_iam_role.batch_service[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# Compute environment: scales to zero when no job is queued
resource "aws_batch_compute_environment" "mujoco" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-mujoco-compute"
  type  = "MANAGED"

  service_role = aws_iam_role.batch_service[0].arn

  compute_resources {
    type               = "EC2"
    instance_role      = aws_iam_instance_profile.batch[0].arn
    instance_type      = [var.batch_instance_type]
    max_vcpus          = var.batch_max_vcpus
    min_vcpus          = 0
    security_group_ids = [aws_security_group.batch[0].id]
    subnets            = local.foundation_subnet_ids

    ec2_configuration {
      image_type = "ECS_AL2023_NVIDIA"
    }

    # Provider default_tags do not reach instances that Batch launches
    tags = {
      Name          = "${local.prefix}-mujoco-batch"
      Project       = var.project_name
      Environment   = var.environment
      Component     = "mujoco"
      "auto-delete" = "no"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "mujoco"
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service]
}

# Job queue
resource "aws_batch_job_queue" "mujoco" {
  count    = var.enable_batch ? 1 : 0
  name     = "${local.prefix}-mujoco-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.mujoco[0].arn
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "mujoco"
  }
}

# Job definition: single node, 1 GPU. Env defaults match
# containers/mujoco/train_entrypoint.py; override them per job at submit time.
# CHECKPOINT_PREFIX is left unset so each job writes checkpoints/<job_id>.
resource "aws_batch_job_definition" "mujoco" {
  count                 = var.enable_batch ? 1 : 0
  name                  = "${local.prefix}-mujoco-train"
  type                  = "container"
  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image      = "${data.aws_ssm_parameter.mujoco_ecr.value}:latest"
    command    = ["batch-train"]
    jobRoleArn = aws_iam_role.batch_job[0].arn
    resourceRequirements = [
      { type = "GPU", value = "1" },
      { type = "VCPU", value = tostring(var.batch_job_vcpus) },
      { type = "MEMORY", value = tostring(var.batch_job_memory_mib) }
    ]
    environment = [
      { name = "ENV_NAME", value = "CartpoleBalance" },
      { name = "NUM_ENVS", value = "2048" },
      { name = "NUM_TIMESTEPS", value = "5000000" },
      { name = "SEED", value = "0" },
      { name = "CHECKPOINT_BUCKET", value = local.batch_checkpoints_bucket }
    ]
  })

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = var.batch_job_timeout_seconds
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "mujoco"
  }
}

# Outputs (conditional)
output "batch_job_queue" {
  description = "Batch job queue name for MuJoCo RL training"
  value       = var.enable_batch ? aws_batch_job_queue.mujoco[0].name : null
}

output "batch_job_definition" {
  description = "Batch job definition name for MuJoCo RL training"
  value       = var.enable_batch ? aws_batch_job_definition.mujoco[0].name : null
}
