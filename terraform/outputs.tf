output "cloudfront_url" {
  description = "CloudFront distribution URL"
  value       = module.cloudfront.distribution_url
}

output "frontend_bucket" {
  description = "S3 bucket for frontend static assets"
  value       = module.s3.bucket_id
}

output "dynamodb_table_name" {
  description = "DynamoDB variants table name"
  value       = module.dynamodb.table_name
}

output "api_gateway_url" {
  description = "API Gateway endpoint URL"
  value       = module.api_gateway.api_url
}

output "sqs_queue_url" {
  description = "SQS processing queue URL"
  value       = module.sqs.queue_url
}

output "worker_function_name" {
  description = "Worker Lambda function name"
  value       = module.lambda_worker.function_name
}
