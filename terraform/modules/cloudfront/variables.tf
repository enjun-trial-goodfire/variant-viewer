variable "app_name" {
  type = string
}

variable "s3_bucket_domain_name" {
  description = "S3 bucket regional domain name for origin"
  type        = string
}

variable "s3_bucket_id" {
  description = "S3 bucket ID for OAC policy"
  type        = string
}

variable "api_gateway_url" {
  description = "API Gateway invoke URL for API origin"
  type        = string
}
