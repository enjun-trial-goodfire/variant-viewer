# ── Dead letter queue ──────────────────────────────────────────────────

resource "aws_sqs_queue" "processing_dlq" {
  name                        = "${var.app_name}-processing-dlq.fifo"
  fifo_queue                  = true
  content_based_deduplication = true
  message_retention_seconds   = 1209600 # 14 days
}

# ── Processing queue ──────────────────────────────────────────────────

resource "aws_sqs_queue" "processing" {
  name                       = "${var.app_name}-processing.fifo"
  fifo_queue                 = true
  visibility_timeout_seconds = 900   # 15 min — must exceed worker Lambda timeout (840s)
  message_retention_seconds  = 86400 # 1 day
  receive_message_wait_time_seconds = 20 # long-poll

  # Explicit MessageDeduplicationId per message (not content-based)
  content_based_deduplication = false

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.processing_dlq.arn
    maxReceiveCount     = 3
  })
}
