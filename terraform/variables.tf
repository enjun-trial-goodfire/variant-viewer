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
