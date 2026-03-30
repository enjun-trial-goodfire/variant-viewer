"""API Lambda handler for variant-viewer.

Routes:
    GET /variants/search?q=...             → DynamoDB GSI Query (begins_with on gene)
    GET /variants/{id}                     → DynamoDB GetItem by variant_id

Environment variables:
    TABLE_NAME  — DynamoDB table name (set by Terraform via Lambda env config)

Response formats:
    /variants/search  → [{v, l, s, c}, ...]  matching frontend renderSearchResults() shape
    /variants/{id}    → full variant JSON record (same shape as build.py output)

This handler is deployed behind API Gateway HTTP API. The event format is
API Gateway v2 (payload format 2.0).
"""

import json
import os
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

TABLE_NAME = os.environ["TABLE_NAME"]
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)

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

    # Map variant_id back to id for frontend compatibility
    item["id"] = item.pop("variant_id")
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

    # GET /variants/{id}
    if path.startswith("/variants/") and path != "/variants/search":
        variant_id = path[len("/variants/"):]
        if not variant_id:
            return json_response(400, {"error": "Missing variant ID"})
        return handle_get_variant(variant_id)

    return json_response(404, {"error": "Not found"})
