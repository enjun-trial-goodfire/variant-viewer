variable "app_name" {
  type = string
}

variable "dynamodb_table" {
  description = "DynamoDB table name for environment variable"
  type        = string
}

variable "dynamodb_arn" {
  description = "DynamoDB table ARN for IAM policy"
  type        = string
}

variable "sqs_queue_url" {
  description = "SQS FIFO queue URL for environment variable"
  type        = string
}

variable "sqs_queue_arn" {
  description = "SQS FIFO queue ARN for IAM policy and event source mapping"
  type        = string
}

variable "anthropic_secret_arn" {
  description = "ARN of the Secrets Manager secret containing the Anthropic API key"
  type        = string
}
