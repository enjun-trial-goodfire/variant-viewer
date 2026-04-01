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
  description = "SQS FIFO queue URL for processing requests (empty = disabled)"
  type        = string
  default     = ""
}

variable "sqs_queue_arn" {
  description = "SQS FIFO queue ARN for IAM policy (empty = no SQS permissions)"
  type        = string
  default     = ""
}
