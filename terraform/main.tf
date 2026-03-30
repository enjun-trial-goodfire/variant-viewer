terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "variant-viewer-terraform-state"
    key    = "variant-viewer/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Step 2: Static frontend hosting ──────────────────────────────────────

module "s3" {
  source   = "./modules/s3"
  app_name = var.app_name
}

module "cloudfront" {
  source                = "./modules/cloudfront"
  app_name              = var.app_name
  s3_bucket_domain_name = module.s3.bucket_regional_domain_name
  s3_bucket_id          = module.s3.bucket_id
}

# ── Step 3: DynamoDB ─────────────────────────────────────────────────────

module "dynamodb" {
  source   = "./modules/dynamodb"
  app_name = var.app_name
}

# ── Steps 5-6: Lambda, API Gateway (uncomment when ready) ───────────────
#
# module "lambda_api" {
#   source         = "./modules/lambda_api"
#   app_name       = var.app_name
#   dynamodb_table = module.dynamodb.table_name
#   dynamodb_arn   = module.dynamodb.table_arn
# }
#
# module "api_gateway" {
#   source               = "./modules/api_gateway"
#   app_name             = var.app_name
#   lambda_invoke_arn    = module.lambda_api.invoke_arn
#   lambda_function_name = module.lambda_api.function_name
# }
