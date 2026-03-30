variable "app_name" {
  type = string
}

variable "s3_bucket_domain_name" {
  description = "S3 bucket regional domain name for origin"
  type        = string
}

variable "s3_bucket_id" {
  description = "S3 bucket ID for bucket policy"
  type        = string
}
