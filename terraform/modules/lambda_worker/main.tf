# ── IAM role ──────────────────────────────────────────────────────────

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "worker" {
  name               = "${var.app_name}-worker-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

# CloudWatch Logs
resource "aws_iam_role_policy_attachment" "worker_logs" {
  role       = aws_iam_role.worker.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# DynamoDB: read full records + write processing status and results
data "aws_iam_policy_document" "dynamodb_readwrite" {
  statement {
    actions = [
      "dynamodb:GetItem",
      "dynamodb:UpdateItem",
    ]
    resources = [var.dynamodb_arn]
  }
}

resource "aws_iam_role_policy" "dynamodb_readwrite" {
  name   = "dynamodb-readwrite"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.dynamodb_readwrite.json
}

# SQS: receive and delete messages
data "aws_iam_policy_document" "sqs_consume" {
  statement {
    actions = [
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
    ]
    resources = [var.sqs_queue_arn]
  }
}

resource "aws_iam_role_policy" "sqs_consume" {
  name   = "sqs-consume"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.sqs_consume.json
}

# Secrets Manager: read Anthropic API key
data "aws_iam_policy_document" "secrets_read" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.anthropic_secret_arn]
  }
}

resource "aws_iam_role_policy" "secrets_read" {
  name   = "secrets-read"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.secrets_read.json
}

# ── Lambda package ────────────────────────────────────────────────────
# Bundle handler + prompt dependencies from their canonical repo locations.
# No file duplication — Terraform pulls from source at plan/apply time.

data "archive_file" "worker_zip" {
  type        = "zip"
  output_path = "${path.module}/lambda_worker.zip"

  # Worker handler
  source {
    content  = file("${path.module}/../../../lambdas/worker/handler.py")
    filename = "handler.py"
  }

  # Prompt building chain: prompts.py → constants.py, display.py
  source {
    content  = file("${path.module}/../../../prompts.py")
    filename = "prompts.py"
  }
  source {
    content  = file("${path.module}/../../../constants.py")
    filename = "constants.py"
  }
  source {
    content  = file("${path.module}/../../../display.py")
    filename = "display.py"
  }

  # Data file used by display.py → curated_group(quality_file=Path("head_quality.json"))
  source {
    content  = file("${path.module}/../../../head_quality.json")
    filename = "head_quality.json"
  }
}

# ── Lambda function ───────────────────────────────────────────────────

resource "aws_lambda_function" "worker" {
  function_name    = "${var.app_name}-worker"
  role             = aws_iam_role.worker.arn
  handler          = "handler.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.worker_zip.output_path
  source_code_hash = data.archive_file.worker_zip.output_base64sha256
  timeout          = 840 # 14 minutes (under Lambda 15-min max)
  memory_size      = 512

  reserved_concurrent_executions = 10 # Cap parallel Claude API calls

  environment {
    variables = {
      TABLE_NAME           = var.dynamodb_table
      ANTHROPIC_SECRET_ARN = var.anthropic_secret_arn
    }
  }
}

# ── SQS trigger ───────────────────────────────────────────────────────

resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn = var.sqs_queue_arn
  function_name    = aws_lambda_function.worker.arn
  batch_size       = 1
  enabled          = true
}
