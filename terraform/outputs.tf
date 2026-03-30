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
