output "mujoco_ecr_uri" {
  description = "ECR URI for the MuJoCo container"
  value       = data.aws_ssm_parameter.mujoco_ecr.value
  sensitive   = true
}

output "codebuild_project" {
  description = "CodeBuild project name for MuJoCo container"
  value       = aws_codebuild_project.mujoco.name
}
