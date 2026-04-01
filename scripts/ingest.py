"""Bulk ingest clean parquet into DynamoDB.

Reads the output of transform.py (one parquet file with all variants) and
writes each row to DynamoDB. Column names in the parquet match the frontend
type exactly, with two DynamoDB-specific aliases:
  - gene_name → gene (GSI partition key)
  - score_pathogenic → score (GSI projected attribute for sort)

Usage:
    python scripts/ingest.py builds/clean.parquet
    python scripts/ingest.py builds/clean.parquet --wipe
    python scripts/ingest.py builds/clean.parquet --table my-table --workers 16
"""

import argparse
import json
import sys
import threading
import time
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

TABLE_DEFAULT = "variant-viewer-variants"
REGION_DEFAULT = "us-east-1"
WORKERS_DEFAULT = 8

# DynamoDB GSI expects these names
DYNAMO_RENAMES = {"gene_name": "gene", "score_pathogenic": "score"}


def to_dynamo(row: dict) -> dict:
    """Convert a parquet row dict to a DynamoDB item."""
    item = {}
    for k, v in row.items():
        if v is None:
            continue
        # Rename for DynamoDB GSI compatibility
        key = DYNAMO_RENAMES.get(k, k)
        # DynamoDB requires Decimal for numbers
        if isinstance(v, float):
            if v != v:  # NaN
                continue
            item[key] = Decimal(str(round(v, 6)))
        elif isinstance(v, int):
            item[key] = v
        else:
            item[key] = v
    item["processing_status"] = "not_started"
    return item


def wipe_table(table):
    """Delete all items from the table."""
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
            print(f"  deleted {deleted:,}...")
        if "LastEvaluatedKey" not in response:
            break
        scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
    print(f"  deleted {deleted:,} total")


# ── Parallel ingestion ───────────────────────────────────────────────

_lock = threading.Lock()
_written = 0
_errors = 0
_failed = []


def _ingest_chunk(rows: list[dict], table_name: str, region: str):
    global _written, _errors
    import boto3
    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    local_ok, local_err, local_failed = 0, 0, []

    with table.batch_writer() as batch:
        for row in rows:
            try:
                batch.put_item(Item=to_dynamo(row))
                local_ok += 1
            except Exception as e:
                local_err += 1
                local_failed.append({"variant_id": row.get("variant_id", "?"), "error": str(e)})

    with _lock:
        _written += local_ok
        _errors += local_err
        _failed.extend(local_failed)


def main():
    parser = argparse.ArgumentParser(description="Ingest clean parquet into DynamoDB")
    parser.add_argument("parquet", type=Path, help="Path to clean.parquet")
    parser.add_argument("--table", default=TABLE_DEFAULT)
    parser.add_argument("--region", default=REGION_DEFAULT)
    parser.add_argument("--workers", type=int, default=WORKERS_DEFAULT)
    parser.add_argument("--wipe", action="store_true", help="Delete all existing items first")
    parser.add_argument("--limit", type=int, default=None, help="Ingest only first N rows (for testing)")
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"Not found: {args.parquet}")
        sys.exit(1)

    # Read parquet
    print(f"Reading {args.parquet}...")
    df = pl.read_parquet(args.parquet)
    if args.limit:
        df = df.head(args.limit)
    rows = df.to_dicts()
    total = len(rows)
    print(f"  {total:,} variants, {df.width} columns")

    # Connect
    import boto3
    table = boto3.resource("dynamodb", region_name=args.region).Table(args.table)
    table.load()
    print(f"Target: {args.table} ({table.item_count:,} items)")

    if args.wipe:
        wipe_table(table)

    # Split into chunks
    chunk_size = max(1, (total + args.workers - 1) // args.workers)
    chunks = [rows[i:i + chunk_size] for i in range(0, total, chunk_size)]
    print(f"Ingesting with {len(chunks)} workers...\n")

    global _written, _errors, _failed
    _written, _errors, _failed = 0, 0, []
    t0 = time.time()

    # Progress
    stop = threading.Event()
    def progress():
        while not stop.wait(5):
            w = _written
            elapsed = time.time() - t0
            rate = w / elapsed if elapsed > 0 else 0
            pct = (w + _errors) / total * 100
            print(f"  {w:,} written, {_errors:,} errors ({pct:.1f}%, {rate:.0f}/s)")
    threading.Thread(target=progress, daemon=True).start()

    # Run
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(_ingest_chunk, chunk, args.table, args.region) for chunk in chunks]
        for f in as_completed(futures):
            f.result()

    stop.set()
    elapsed = time.time() - t0
    print(f"\nDone: {_written:,} written, {_errors:,} errors in {elapsed:.1f}s ({_written/elapsed:.0f}/s)")

    if _failed:
        fail_path = args.parquet.parent / "ingest_failures.json"
        json.dump(_failed, open(fail_path, "w"), indent=2)
        print(f"Failures: {fail_path}")


if __name__ == "__main__":
    main()
