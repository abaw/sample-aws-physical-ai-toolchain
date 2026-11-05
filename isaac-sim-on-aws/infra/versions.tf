terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # The account's cleanup deletes resources older than 7 days unless they
  # are tagged auto-delete=no.
  default_tags {
    tags = {
      "auto-delete" = "no"
    }
  }
}
