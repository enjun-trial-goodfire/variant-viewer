variable "app_name" {
  type = string
}

variable "lambda_invoke_arn" {
  description = "Lambda function invoke ARN for API Gateway integration"
  type        = string
}

variable "lambda_function_name" {
  description = "Lambda function name for permission grant"
  type        = string
}
