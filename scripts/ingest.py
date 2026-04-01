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
    python scripts/ingest.py /path/to/variants/dir --workers 16
    python scripts/ingest.py /path/to/variants/dir --table my-table --region us-west-2

Requirements:
    boto3 (pip install boto3 or available in the AWS Lambda runtime)
    AWS credentials configured (aws configure, env vars, or IAM role)
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import boto3

TABLE_DEFAULT = "variant-viewer-variants"
REGION_DEFAULT = "us-east-1"
WORKERS_DEFAULT = 8


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


# ── Parallel ingestion ───────────────────────────────────────────────

_lock = threading.Lock()
_written = 0
_errors = 0
_failed_files = []
_t0 = 0.0


def _process_chunk(chunk, table_name, region):
    """Process a chunk of files in a single thread with its own batch_writer."""
    global _written, _errors

    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)
    local_written = 0
    local_errors = 0
    local_failed = []

    with table.batch_writer() as batch:
        for f in chunk:
            try:
                data = json.loads(f.read_bytes())
                variant_id = data.pop("id")
                gene = data.get("gene", "")
                item = convert_floats(data)
                item["variant_id"] = variant_id
                item["gene"] = gene
                item["processing_status"] = "not_started"
                batch.put_item(Item=item)
                local_written += 1
            except Exception as e:
                local_errors += 1
                local_failed.append({"file": f.name, "error": str(e)})

    with _lock:
        _written += local_written
        _errors += local_errors
        _failed_files.extend(local_failed)


def ingest(source_dir: Path, table_name: str, region: str, wipe: bool, workers: int):
    """Load variant JSONs from source_dir into DynamoDB using parallel workers."""
    global _written, _errors, _failed_files, _t0
    _written = 0
    _errors = 0
    _failed_files = []

    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {source_dir}")
        sys.exit(1)

    total = len(json_files)
    print(f"Found {total:,} variant files in {source_dir}")

    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)
    table.load()
    print(f"Target table: {table_name} ({table.item_count:,} items reported)")

    if wipe:
        wipe_table(table)

    # Split files into chunks, one per worker
    chunk_size = (total + workers - 1) // workers
    chunks = [json_files[i:i + chunk_size] for i in range(0, total, chunk_size)]
    print(f"Ingesting with {len(chunks)} workers ({chunk_size:,} files each)...\n")

    _t0 = time.time()

    # Progress reporter in a background thread
    stop_progress = threading.Event()

    def report_progress():
        while not stop_progress.wait(5):
            elapsed = time.time() - _t0
            w = _written
            e = _errors
            done = w + e
            if done == 0:
                continue
            rate = w / elapsed
            remaining = (total - done) / rate if rate > 0 else 0
            pct = done / total * 100
            print(f"  {w:,} written, {e:,} errors ({pct:.1f}%, {rate:.0f}/s, ~{remaining/60:.0f}m remaining)")

    progress_thread = threading.Thread(target=report_progress, daemon=True)
    progress_thread.start()

    # Run workers
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(_process_chunk, chunk, table_name, region) for chunk in chunks]
        for f in as_completed(futures):
            f.result()  # Raise any unhandled exceptions

    stop_progress.set()
    elapsed = time.time() - _t0
    rate = _written / elapsed if elapsed > 0 else 0

    print(f"\nDone: {_written:,} written, {_errors:,} errors in {elapsed:.1f}s ({rate:.0f}/s)")

    # Write failures to file for retry
    if _failed_files:
        failed_path = source_dir / "ingest_failures.json"
        with open(failed_path, "w") as fh:
            json.dump(_failed_files, fh, indent=2)
        print(f"\nFailed records written to {failed_path}")
        print("First 10 failures:")
        for entry in _failed_files[:10]:
            print(f"  {entry['file']}: {entry['error']}")


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest variant JSONs into DynamoDB")
    parser.add_argument("source_dir", type=Path, help="Directory containing per-variant JSON files")
    parser.add_argument("--table", default=TABLE_DEFAULT, help=f"DynamoDB table name (default: {TABLE_DEFAULT})")
    parser.add_argument("--region", default=REGION_DEFAULT, help=f"AWS region (default: {REGION_DEFAULT})")
    parser.add_argument("--workers", type=int, default=WORKERS_DEFAULT, help=f"Parallel worker threads (default: {WORKERS_DEFAULT})")
    parser.add_argument("--wipe", action="store_true", help="Delete all existing items before ingesting")
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        print(f"Source directory not found: {args.source_dir}")
        sys.exit(1)

    ingest(args.source_dir, args.table, args.region, args.wipe, args.workers)


if __name__ == "__main__":
    main()
