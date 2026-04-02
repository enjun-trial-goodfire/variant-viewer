"""API Lambda handler for variant-viewer.

Routes:
    GET /variants/search?q=...             → DynamoDB GSI Query (begins_with on gene)
    GET /variants/{id}                     → DynamoDB GetItem by variant_id
    GET /variants/{id}/analysis            → Get-or-create interpretation (202 polling pattern)

Environment variables:
    TABLE_NAME     — DynamoDB table name (set by Terraform via Lambda env config)
    SQS_QUEUE_URL  — SQS FIFO queue URL for processing requests (optional, empty = disabled)

Response formats:
    /variants/search        → [{v, l, s, c}, ...]  matching frontend renderSearchResults() shape
    /variants/{id}          → full variant JSON record (same shape as build.py output)
    /variants/{id}/analysis → 200 {status, result} or 202 {status, retry_after}

This handler is deployed behind API Gateway HTTP API. The event format is
API Gateway v2 (payload format 2.0).
"""

import json
import os
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

TABLE_NAME = os.environ["TABLE_NAME"]
SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "")

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)
sqs = boto3.client("sqs") if SQS_QUEUE_URL else None

SEARCH_LIMIT = 30


class DecimalEncoder(json.JSONEncoder):
    """Convert DynamoDB Decimals back to int/float for JSON response."""

    def default(self, o):
        if isinstance(o, Decimal):
            return int(o) if o == int(o) else float(o)
        return super().default(o)


def json_response(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, cls=DecimalEncoder),
    }


def handle_get_variant(variant_id):
    """GET /variants/{id} → full variant record."""
    resp = table.get_item(Key={"variant_id": variant_id})
    item = resp.get("Item")
    if not item:
        return json_response(404, {"error": "Variant not found"})

    # DynamoDB GSI aliases: ingest renames gene_name→gene, score_pathogenic→score.
    # Svelte frontend expects the original names, so add them back.
    item.setdefault("gene_name", item.get("gene"))
    item.setdefault("score_pathogenic", item.get("score"))

    return json_response(200, item)


def handle_search(query):
    """GET /variants/search?q=... → search by gene name prefix via GSI."""
    q = query.strip().upper()
    if len(q) < 2:
        return json_response(200, [])

    resp = table.query(
        IndexName="gene-index",
        KeyConditionExpression=Key("gene").eq(q),
        Limit=SEARCH_LIMIT,
    )
    results = resp.get("Items", [])

    # If exact match returned few results, also try prefix matches on other genes
    if len(results) < SEARCH_LIMIT:
        # Scan for genes that start with the query (e.g., "BRC" → "BRCA1", "BRCA2")
        # DynamoDB GSI query requires exact partition key, so we do a limited scan
        # with a filter for prefix matching across different gene names
        scan_resp = table.scan(
            IndexName="gene-index",
            FilterExpression="begins_with(gene, :prefix)",
            ExpressionAttributeValues={":prefix": q},
            Limit=1000,  # scan limit (rows examined, not returned)
        )
        prefix_items = scan_resp.get("Items", [])

        # Merge, dedup by variant_id
        seen = {r["variant_id"] for r in results}
        for item in prefix_items:
            if item["variant_id"] not in seen:
                results.append(item)
                seen.add(item["variant_id"])

    # Sort by score descending, limit to SEARCH_LIMIT
    results.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    results = results[:SEARCH_LIMIT]

    # Map to frontend shape: {v, l, s, c}
    return json_response(200, [
        {
            "v": r["variant_id"],
            "l": r.get("label", ""),
            "s": r.get("score", 0),
            "c": r.get("consequence", ""),
        }
        for r in results
    ])


def handle_get_analysis(variant_id):
    """GET /variants/{id}/analysis → get-or-create interpretation.

    Returns:
        200 + {status: "complete", result: {...}}   — cached result
        202 + {status: "queued", retry_after: 10}   — just enqueued
        202 + {status: "processing", retry_after: 10} — in progress
        404                                         — variant not found
        503                                         — processing not configured
    """
    resp = table.get_item(
        Key={"variant_id": variant_id},
        ProjectionExpression="variant_id, processing_status, processed_result, processing_error",
    )
    item = resp.get("Item")
    if item is None:
        return json_response(404, {"error": "Variant not found"})

    status = item.get("processing_status", "not_started")

    # Already complete → return cached result
    if status == "complete":
        result = item.get("processed_result", {})
        return json_response(200, {"status": "complete", "result": result})

    # Already in progress → tell frontend to poll
    if status in ("pending", "processing"):
        return json_response(202, {"status": "processing", "retry_after": 10})

    # not_started or failed → enqueue for processing
    if not sqs or not SQS_QUEUE_URL:
        return json_response(503, {"error": "Processing not configured"})

    # Atomic conditional update: only transition from not_started/failed/missing → pending
    try:
        table.update_item(
            Key={"variant_id": variant_id},
            UpdateExpression="SET processing_status = :pending",
            ConditionExpression="attribute_not_exists(processing_status) OR processing_status IN (:not_started, :failed)",
            ExpressionAttributeValues={
                ":pending": "pending",
                ":not_started": "not_started",
                ":failed": "failed",
            },
        )
    except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
        # Race: another request already moved it to pending/processing
        return json_response(202, {"status": "processing", "retry_after": 10})

    # Send to SQS FIFO
    sqs.send_message(
        QueueUrl=SQS_QUEUE_URL,
        MessageBody=json.dumps({"variant_id": variant_id}),
        MessageGroupId=variant_id,
        MessageDeduplicationId=variant_id,
    )

    return json_response(202, {"status": "queued", "retry_after": 10})


def handler(event, context):
    """Lambda entry point. Routes API Gateway v2 events."""
    path = event.get("rawPath", "")
    method = event.get("requestContext", {}).get("http", {}).get("method", "")

    if method != "GET":
        return json_response(405, {"error": "Method not allowed"})

    # GET /variants/search?q=...
    if path == "/variants/search":
        params = event.get("queryStringParameters") or {}
        query = params.get("q", "")
        if not query:
            return json_response(400, {"error": "Missing q parameter"})
        return handle_search(query)

    # GET /variants/{id}/analysis — must come before the generic /variants/{id} catch-all
    if path.endswith("/analysis") and path.startswith("/variants/"):
        variant_id = path[len("/variants/"):-len("/analysis")]
        if not variant_id:
            return json_response(400, {"error": "Missing variant ID"})
        return handle_get_analysis(variant_id)

    # GET /variants/{id}
    if path.startswith("/variants/") and path != "/variants/search":
        variant_id = path[len("/variants/"):]
        if not variant_id:
            return json_response(400, {"error": "Missing variant ID"})
        return handle_get_variant(variant_id)

    return json_response(404, {"error": "Not found"})
