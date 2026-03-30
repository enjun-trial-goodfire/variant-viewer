"""Bulk ingest variant JSONs into DynamoDB.

Overview:
    Reads pre-built per-variant JSON files and writes them to the
    variant-viewer-variants DynamoDB table. This is the bridge between
    the data pipeline (build.py on the cluster) and the AWS-hosted app.

Source data:
    build.py reads probe scores + annotations from NFS, runs GPU computation,
    and writes one JSON file per variant to a staging directory. These files
    are typically found at:
        /mnt/polished-lake/artifacts/.../probe_webapp_build/variants/

    Each JSON (~18KB) contains the full variant record: id, gene, score, label,
    consequence, disruption scores, effect scores, neighbors, VEP annotations,
    gnomAD frequencies, ClinVar metadata, etc. The schema is documented in
    SCHEMA.md at the repo root.

Key mapping:
    - JSON "id" field (e.g., "chr10:100042514:C:T") → DynamoDB "variant_id" (partition key)
    - JSON "gene" field (e.g., "BRCA1") → DynamoDB "gene" (GSI partition key for search)
    - All other fields are stored as-is (DynamoDB is schemaless for non-key attributes)
    - "processing_status" is added as "not_started" on each item (for future lazy analysis)

Wipe behavior:
    Use --wipe for full dataset replacement (e.g., switching from test set to
    production). This deletes all existing items before ingesting. DynamoDB
    PutItem overwrites on key collision, so --wipe is only needed when the new
    dataset is missing variants that were in the old set (otherwise stale
    records would remain).

Usage:
    python scripts/ingest.py /path/to/variants/dir
    python scripts/ingest.py /path/to/variants/dir --wipe
    python scripts/ingest.py /path/to/variants/dir --table my-table --region us-west-2

Requirements:
    boto3 (pip install boto3 or available in the AWS Lambda runtime)
    AWS credentials configured (aws configure, env vars, or IAM role)
"""

import argparse
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import boto3
from boto3.dynamodb.types import TypeSerializer

TABLE_DEFAULT = "variant-viewer-variants"
REGION_DEFAULT = "us-east-1"
BATCH_SIZE = 25  # DynamoDB batch_write_item limit


def convert_floats(obj):
    """Recursively convert floats to Decimals (DynamoDB requirement)."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_floats(v) for v in obj]
    return obj


def wipe_table(table):
    """Delete all items from the table via scan + batch delete."""
    print("Wiping existing items...")
    deleted = 0
    scan_kwargs = {"ProjectionExpression": "variant_id"}

    while True:
        response = table.scan(**scan_kwargs)
        items = response.get("Items", [])
        if not items:
            break

        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={"variant_id": item["variant_id"]})
                deleted += 1

        if deleted % 10000 == 0:
            print(f"  deleted {deleted:,} items...")

        if "LastEvaluatedKey" not in response:
            break
        scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

    print(f"  deleted {deleted:,} items total")
    return deleted


def ingest(source_dir: Path, table_name: str, region: str, wipe: bool):
    """Load variant JSONs from source_dir into DynamoDB."""
    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(json_files):,} variant files in {source_dir}")

    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    # Verify table exists
    table.load()
    print(f"Target table: {table_name} ({table.item_count:,} items reported)")

    if wipe:
        wipe_table(table)

    # Batch write
    written = 0
    errors = 0
    t0 = time.time()

    with table.batch_writer() as batch:
        for f in json_files:
            try:
                data = json.loads(f.read_bytes())
                variant_id = data.pop("id")
                gene = data.get("gene", "")
                item = convert_floats(data)
                item["variant_id"] = variant_id
                item["gene"] = gene
                item["processing_status"] = "not_started"
                batch.put_item(Item=item)
                written += 1
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  ERROR on {f.name}: {e}")

            if written % 10000 == 0 and written > 0:
                elapsed = time.time() - t0
                rate = written / elapsed
                print(f"  {written:,} written ({rate:.0f}/s)")

    elapsed = time.time() - t0
    print(f"\nDone: {written:,} written, {errors:,} errors in {elapsed:.1f}s ({written/elapsed:.0f}/s)")


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest variant JSONs into DynamoDB")
    parser.add_argument("source_dir", type=Path, help="Directory containing per-variant JSON files")
    parser.add_argument("--table", default=TABLE_DEFAULT, help=f"DynamoDB table name (default: {TABLE_DEFAULT})")
    parser.add_argument("--region", default=REGION_DEFAULT, help=f"AWS region (default: {REGION_DEFAULT})")
    parser.add_argument("--wipe", action="store_true", help="Delete all existing items before ingesting")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Source directory not found: {args.source_dir}")
        sys.exit(1)

    ingest(args.source_dir, args.table, args.region, args.wipe)


if __name__ == "__main__":
    main()
