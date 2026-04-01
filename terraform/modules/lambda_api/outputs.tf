output "invoke_arn" {
  description = "Lambda invoke ARN for API Gateway integration"
  value       = aws_lambda_function.api.invoke_arn
}

output "function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.api.function_name
}
