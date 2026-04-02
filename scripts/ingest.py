"""Ingest variants from DuckDB → DynamoDB.

Reads the full variant table from DuckDB (which includes neighbors from
the build step), converts each row to a DynamoDB item, and batch-writes
with parallel workers.

Usage:
    uv run python scripts/ingest.py                          # full ingest, 128 workers
    uv run python scripts/ingest.py --limit 100              # test with 100 rows
    uv run python scripts/ingest.py --workers 16 --limit 500 # slower, for debugging
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import duckdb

# DynamoDB GSI expects 'gene' and 'score', not the parquet names
RENAMES = {"gene_name": "gene", "score_pathogenic": "score"}
MAX_KEY_BYTES = 2048  # DynamoDB partition key size limit


def to_dynamo_item(row: dict) -> dict | None:
    """Convert a DuckDB row dict to a DynamoDB item. Returns None if variant_id too long."""
    vid = row.get("variant_id")
    if not vid or len(vid.encode("utf-8")) > MAX_KEY_BYTES:
        return None
    item = {"processing_status": "not_started"}
    for k, v in row.items():
        if v is None:
            continue
        if isinstance(v, float):
            if v != v:  # NaN
                continue
            v = Decimal(str(round(v, 6)))
        item[RENAMES.get(k, k)] = v
    return item


# ── Worker state ─────────────────────────────────────────────────────

_lock = threading.Lock()
_written = 0
_errors = 0
_skipped = 0
_error_samples: list[str] = []


def _ingest_chunk(chunk: list[dict], table_name: str, region: str):
    """Process a chunk of row dicts in a single thread."""
    global _written, _errors, _skipped
    import boto3
    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    ok, err, skip = 0, 0, 0
    errors = []
    with table.batch_writer() as batch:
        for row in chunk:
            item = to_dynamo_item(row)
            if item is None:
                skip += 1
                continue
            try:
                batch.put_item(Item=item)
                ok += 1
            except Exception as e:
                err += 1
                if len(errors) < 3:
                    errors.append(f"{row.get('variant_id', '?')}: {e}")
    with _lock:
        _written += ok
        _errors += err
        _skipped += skip
        _error_samples.extend(errors)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    global _written, _errors, _skipped, _error_samples

    parser = argparse.ArgumentParser(description="Ingest DuckDB → DynamoDB")
    parser.add_argument("--db", type=Path, default=Path("builds/variants.duckdb"))
    parser.add_argument("--table", default="variant-viewer-variants")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Ingest only first N rows")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"Not found: {args.db}")
        sys.exit(1)

    # Read all rows from DuckDB
    conn = duckdb.connect(str(args.db), read_only=True)
    limit_clause = f"LIMIT {args.limit}" if args.limit else ""
    print(f"Reading from {args.db}...")
    result = conn.execute(f"SELECT * FROM variants {limit_clause}")
    cols = [desc[0] for desc in result.description]
    raw_rows = result.fetchall()
    rows = [dict(zip(cols, row)) for row in raw_rows]
    del raw_rows  # free memory
    total = len(rows)
    has_neighbors = sum(1 for r in rows[:100] if r.get("neighbors") is not None)
    print(f"  {total:,} rows, {len(cols)} columns, neighbors in sample: {has_neighbors}/100")

    # Split into chunks for workers
    chunk_size = max(1, (total + args.workers - 1) // args.workers)
    chunks = [rows[i:i + chunk_size] for i in range(0, total, chunk_size)]
    print(f"  {len(chunks)} chunks of ~{chunk_size:,} rows, {args.workers} workers\n")

    _written, _errors, _skipped, _error_samples = 0, 0, 0, []
    t0 = time.time()

    # Progress thread
    stop = threading.Event()
    def progress():
        while not stop.wait(10):
            w = _written
            elapsed = time.time() - t0
            rate = w / elapsed if elapsed > 0 else 0
            remaining = (total - w - _errors - _skipped) / rate / 60 if rate > 0 else 0
            print(f"  {w:,}/{total:,} ({w/total*100:.1f}%, {rate:.0f}/s, ~{remaining:.0f}m left, {_errors} errors)")
    threading.Thread(target=progress, daemon=True).start()

    # Run
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_ingest_chunk, chunk, args.table, args.region) for chunk in chunks]
        for f in as_completed(futures):
            f.result()

    stop.set()
    elapsed = time.time() - t0
    print(f"\nDone: {_written:,} written, {_errors} errors, {_skipped} skipped in {elapsed:.0f}s ({_written/elapsed:.0f}/s)")
    if _error_samples:
        print("Sample errors:")
        for e in _error_samples[:5]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
