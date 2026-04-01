output "queue_url" {
  description = "SQS FIFO processing queue URL"
  value       = aws_sqs_queue.processing.url
}

output "queue_arn" {
  description = "SQS FIFO processing queue ARN"
  value       = aws_sqs_queue.processing.arn
}

output "dlq_arn" {
  description = "SQS FIFO dead letter queue ARN"
  value       = aws_sqs_queue.processing_dlq.arn
}
