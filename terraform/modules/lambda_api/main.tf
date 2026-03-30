# ── IAM role for the Lambda function ──────────────────────────────────────

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.app_name}-api-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# CloudWatch Logs
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# DynamoDB read access (GetItem, Query, Scan on table + indexes)
data "aws_iam_policy_document" "dynamodb_read" {
  statement {
    actions = [
      "dynamodb:GetItem",
      "dynamodb:Query",
      "dynamodb:Scan",
    ]
    resources = [
      var.dynamodb_arn,
      "${var.dynamodb_arn}/index/*",
    ]
  }
}

resource "aws_iam_role_policy" "dynamodb_read" {
  name   = "dynamodb-read"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.dynamodb_read.json
}

# ── Lambda function ──────────────────────────────────────────────────────

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../../lambdas/api"
  output_path = "${path.module}/lambda_api.zip"
}

resource "aws_lambda_function" "api" {
  function_name    = "${var.app_name}-api"
  role             = aws_iam_role.lambda.arn
  handler          = "handler.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout          = 10
  memory_size      = 256

  environment {
    variables = {
      TABLE_NAME = var.dynamodb_table
    }
  }
}
