resource "aws_dynamodb_table" "variants" {
  name         = "${var.app_name}-variants"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "variant_id"

  attribute {
    name = "variant_id"
    type = "S"
  }

  attribute {
    name = "gene"
    type = "S"
  }

  # GSI for searching variants by gene name (begins_with prefix search)
  global_secondary_index {
    name            = "gene-index"
    hash_key        = "gene"
    range_key       = "variant_id"
    projection_type = "INCLUDE"
    non_key_attributes = [
      "label",
      "score",
      "consequence",
    ]
  }
}
