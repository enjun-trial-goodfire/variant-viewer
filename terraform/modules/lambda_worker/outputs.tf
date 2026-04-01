output "function_name" {
  description = "Worker Lambda function name"
  value       = aws_lambda_function.worker.function_name
}

output "function_arn" {
  description = "Worker Lambda function ARN"
  value       = aws_lambda_function.worker.arn
}
