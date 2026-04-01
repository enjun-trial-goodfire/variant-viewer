variable "app_name" {
  description = "Application name used as prefix for all resources"
  type        = string
  default     = "variant-viewer"
}

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "anthropic_secret_arn" {
  description = "ARN of the Secrets Manager secret containing the Anthropic API key"
  type        = string
}
