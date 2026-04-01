"""Worker Lambda: process variant interpretation via Claude API.

Trigger: SQS FIFO queue, batch_size=1.
Each message contains {"variant_id": "chr10:100042514:C:T"}.

Flow:
    1. Atomic claim (conditional DynamoDB update: pending → processing)
    2. Build interpretation prompt from variant data (via prompts.py)
    3. Call Claude API via raw HTTP (zero external dependencies)
    4. Store result in DynamoDB, set processing_status = complete
    5. On failure: set processing_status = failed, re-raise for SQS retry

Environment variables:
    TABLE_NAME           — DynamoDB table name
    ANTHROPIC_SECRET_ARN — Secrets Manager ARN for the Anthropic API key
"""

import json
import os
import time
import urllib.error
import urllib.request
from decimal import Decimal

import boto3

from prompts import SYSTEM_PROMPT, build_prompt

# ── Cold-start globals ───────────────────────────────────────────────

TABLE_NAME = os.environ["TABLE_NAME"]
SECRET_ARN = os.environ["ANTHROPIC_SECRET_ARN"]

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)
secrets = boto3.client("secretsmanager")

_api_key = None


def _get_api_key():
    """Fetch API key from Secrets Manager, cached for Lambda execution lifetime."""
    global _api_key
    if _api_key is None:
        resp = secrets.get_secret_value(SecretId=SECRET_ARN)
        _api_key = resp["SecretString"]
    return _api_key


# ── Claude API (raw HTTP, no SDK) ────────────────────────────────────

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-6"

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "mechanism": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "key_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "mechanism", "confidence", "key_evidence"],
    "additionalProperties": False,
}


def call_claude(prompt_text):
    """Call Claude Messages API via urllib. Returns parsed interpretation dict."""
    api_key = _get_api_key()
    body = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 1024,
        "temperature": 0.3,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt_text}],
        "output_config": {
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
    }).encode()

    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    text = next(
        (b["text"] for b in data.get("content", []) if b.get("type") == "text"),
        "{}",
    )
    return json.loads(text)


# ── DynamoDB helpers ─────────────────────────────────────────────────

class DecimalEncoder(json.JSONEncoder):
    """Convert DynamoDB Decimals to int/float for JSON serialization."""

    def default(self, o):
        if isinstance(o, Decimal):
            return int(o) if o == int(o) else float(o)
        return super().default(o)


def _convert_floats(obj):
    """Recursively convert floats to Decimals for DynamoDB writes."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _convert_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_floats(v) for v in obj]
    return obj


def _normalize_disruption(item):
    """Convert [ref, var] disruption pairs to delta scalars if needed.

    Matches the normalization in the API Lambda (lambdas/api/handler.py)
    so prompts.py sees the same data shape regardless of ingestion format.
    """
    disruption = item.get("disruption", {})
    if not disruption:
        return
    sample = next(iter(disruption.values()), None)
    if isinstance(sample, list):
        item["disruption"] = {
            k: round(float(v[1]) - float(v[0]), 4)
            for k, v in disruption.items()
            if isinstance(v, list) and len(v) == 2
            and abs(float(v[1]) - float(v[0])) > 0.01
        }


def claim_variant(variant_id):
    """Atomic claim: pending → processing. Returns True if claimed, False if already taken."""
    try:
        table.update_item(
            Key={"variant_id": variant_id},
            UpdateExpression="SET processing_status = :processing",
            ConditionExpression="processing_status = :pending",
            ExpressionAttributeValues={
                ":processing": "processing",
                ":pending": "pending",
            },
        )
        return True
    except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
        return False


def complete_variant(variant_id, result):
    """Write interpretation result and set status = complete."""
    table.update_item(
        Key={"variant_id": variant_id},
        UpdateExpression=(
            "SET processing_status = :complete, "
            "processed_at = :now, "
            "processed_result = :result"
        ),
        ExpressionAttributeValues={
            ":complete": "complete",
            ":now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ":result": _convert_floats(result),
        },
    )


def fail_variant(variant_id, error_msg):
    """Set status = failed with error info."""
    table.update_item(
        Key={"variant_id": variant_id},
        UpdateExpression=(
            "SET processing_status = :failed, "
            "processed_at = :now, "
            "processing_error = :err"
        ),
        ExpressionAttributeValues={
            ":failed": "failed",
            ":now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ":err": error_msg[:1000],
        },
    )


# ── Handler ──────────────────────────────────────────────────────────

def handler(event, context):
    """Process one SQS message containing a variant_id."""
    for record in event.get("Records", []):
        body = json.loads(record["body"])
        variant_id = body["variant_id"]
        print(f"Processing: {variant_id}")

        # 1. Atomic claim — idempotent guard against duplicate SQS delivery
        if not claim_variant(variant_id):
            print(f"Skipping (already claimed): {variant_id}")
            return

        try:
            # 2. Read full variant record
            resp = table.get_item(Key={"variant_id": variant_id})
            item = resp.get("Item")
            if not item:
                raise ValueError(f"Variant not found in DynamoDB: {variant_id}")

            # 3. Prepare variant data for prompt building
            v = json.loads(json.dumps(dict(item), cls=DecimalEncoder))
            v["id"] = v.pop("variant_id")
            _normalize_disruption(v)

            prompt = build_prompt(v)
            print(f"Prompt length: {len(prompt)} chars")

            # 4. Call Claude API
            result = call_claude(prompt)
            result.update({
                "status": "ok",
                "variant_id": variant_id,
                "model": CLAUDE_MODEL,
                "generated_at": time.time(),
            })
            print(f"Result: confidence={result.get('confidence')}")

            # 5. Store result
            complete_variant(variant_id, result)
            print(f"Complete: {variant_id}")

        except Exception as e:
            print(f"Failed: {variant_id}: {e}")
            fail_variant(variant_id, str(e))
            raise  # Re-raise so SQS retries (up to maxReceiveCount → DLQ)
