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
