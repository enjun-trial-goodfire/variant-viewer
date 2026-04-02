"""Ingest variants from clean.parquet → DynamoDB via S3 ImportTable.

Same-name replacement workflow (no Lambda/IAM changes needed):
    1. Convert parquet → gzipped DynamoDB JSON
    2. Upload to S3
    3. ImportTable → staging table (old table still serving)
    4. Verify staging table (item count, spot-check, GSI query)
    5. Delete old table
    6. ImportTable → original table name (brief downtime ~10-15 min)
    7. Verify final table
    8. Delete staging table + clean up S3

Usage:
    uv run python scripts/ingest.py                     # full pipeline
    uv run python scripts/ingest.py --limit 100         # test with 100 rows
    uv run python scripts/ingest.py --staging-only      # import staging + verify, stop before swap
"""

import argparse
import gzip
import math
import sys
import time
from pathlib import Path

import boto3
import orjson
import polars as pl
from loguru import logger

APP_NAME = "variant-viewer"
S3_BUCKET = f"{APP_NAME}-frontend"
S3_PREFIX = "imports"
TABLE_NAME = f"{APP_NAME}-variants"
STAGING_TABLE = f"{TABLE_NAME}-staging"
REGION = "us-east-1"
MAX_KEY_BYTES = 2048

# GSI expects 'gene' and 'score'; map from parquet canonical names.
GSI_RENAMES = {"gene_name": "gene", "pathogenicity": "score"}


# ── DynamoDB JSON conversion ────────────────────────────────────────


def to_dynamo_value(v: object) -> dict | None:
    """Convert a Python value to a DynamoDB typed-value dict. Returns None to skip."""
    if v is None:
        return None
    if isinstance(v, bool):  # before int — bool is subclass of int
        return {"BOOL": v}
    if isinstance(v, str):
        return {"S": v}
    if isinstance(v, int):
        return {"N": str(v)}
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return {"N": str(round(v, 6))}
    return {"S": str(v)}


def convert_parquet(parquet_path: Path, output_path: Path, limit: int | None) -> int:
    """Convert parquet to gzipped DynamoDB JSON lines. Returns row count."""
    logger.info(f"Reading {parquet_path}")
    df = pl.read_parquet(parquet_path)
    if limit:
        df = df.head(limit)
    logger.info(f"  {df.height:,} rows, {df.width} columns")

    count = 0
    t0 = time.time()
    with gzip.open(output_path, "wb", compresslevel=6) as f:
        for row in df.iter_rows(named=True):
            vid = row.get("variant_id")
            if not vid or len(vid.encode("utf-8")) > MAX_KEY_BYTES:
                continue

            item: dict[str, dict] = {"processing_status": {"S": "not_started"}}
            for k, v in row.items():
                dv = to_dynamo_value(v)
                if dv is not None:
                    item[k] = dv
                    # Add GSI aliases alongside canonical names
                    if k in GSI_RENAMES:
                        item[GSI_RENAMES[k]] = dv

            f.write(orjson.dumps({"Item": item}))
            f.write(b"\n")
            count += 1
            if count % 50_000 == 0:
                logger.info(f"  {count:,}/{df.height:,} rows ({time.time() - t0:.0f}s)")

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  {count:,} rows → {size_mb:.1f} MB gzipped in {time.time() - t0:.0f}s")
    return count


# ── AWS operations ──────────────────────────────────────────────────


def upload_to_s3(local_path: Path, s3_key: str) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    size_mb = local_path.stat().st_size / 1024 / 1024
    logger.info(f"Uploading {size_mb:.1f} MB to s3://{S3_BUCKET}/{s3_key}")
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    logger.info("  Done")


def start_import(table_name: str, s3_prefix: str) -> str:
    """Start ImportTable, return the import ARN."""
    dynamodb = boto3.client("dynamodb", region_name=REGION)
    resp = dynamodb.import_table(
        S3BucketSource={"S3Bucket": S3_BUCKET, "S3KeyPrefix": s3_prefix},
        InputFormat="DYNAMODB_JSON",
        InputCompressionType="GZIP",
        TableCreationParameters={
            "TableName": table_name,
            "AttributeDefinitions": [
                {"AttributeName": "variant_id", "AttributeType": "S"},
                {"AttributeName": "gene", "AttributeType": "S"},
            ],
            "KeySchema": [
                {"AttributeName": "variant_id", "KeyType": "HASH"},
            ],
            "BillingMode": "PAY_PER_REQUEST",
            "GlobalSecondaryIndexes": [{
                "IndexName": "gene-index",
                "KeySchema": [
                    {"AttributeName": "gene", "KeyType": "HASH"},
                    {"AttributeName": "variant_id", "KeyType": "RANGE"},
                ],
                "Projection": {
                    "ProjectionType": "INCLUDE",
                    "NonKeyAttributes": ["label", "score", "consequence"],
                },
            }],
        },
    )
    arn = resp["ImportTableDescription"]["ImportArn"]
    logger.info(f"ImportTable started: {arn}")
    return arn


def wait_for_import(import_arn: str) -> dict:
    """Poll until import completes. Returns final description."""
    dynamodb = boto3.client("dynamodb", region_name=REGION)
    while True:
        desc = dynamodb.describe_import(ImportArn=import_arn)["ImportTableDescription"]
        status = desc["ImportStatus"]
        imported = desc.get("ImportedItemCount", 0)
        errors = desc.get("ErrorCount", 0)
        logger.info(f"  {status} — {imported:,} imported, {errors} errors")

        if status == "COMPLETED":
            return desc
        if status in ("FAILED", "CANCELLED", "CANCELLING"):
            raise RuntimeError(f"Import {status}: {desc.get('FailureCode')} — {desc.get('FailureMessage')}")
        time.sleep(30)


def wait_for_delete(table_name: str) -> None:
    """Wait until a table is fully deleted."""
    dynamodb = boto3.client("dynamodb", region_name=REGION)
    while True:
        try:
            resp = dynamodb.describe_table(TableName=table_name)
            status = resp["Table"]["TableStatus"]
            logger.info(f"  {table_name}: {status}")
            time.sleep(5)
        except dynamodb.exceptions.ResourceNotFoundException:
            logger.info(f"  {table_name}: deleted")
            return


def verify_table(table_name: str, expected_count: int) -> None:
    """Spot-check the table: scan one item, query the GSI."""
    from boto3.dynamodb.conditions import Key

    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(table_name)
    table.reload()
    logger.info(f"  Item count (approx): {table.item_count:,} (expected ~{expected_count:,})")

    resp = table.scan(Limit=1)
    items = resp.get("Items", [])
    assert items, "Table scan returned 0 items"
    sample = items[0]
    for col in ("variant_id", "gene", "score", "processing_status"):
        assert col in sample, f"Sample item missing {col}"
    logger.info(f"  Sample OK: {sample['variant_id']}")

    gene = sample["gene"]
    gsi_resp = table.query(
        IndexName="gene-index",
        KeyConditionExpression=Key("gene").eq(gene),
        Limit=1,
    )
    assert gsi_resp["Count"] > 0, f"GSI query for gene={gene} returned 0"
    logger.info(f"  GSI OK: gene={gene}")


def delete_table(table_name: str) -> None:
    dynamodb = boto3.client("dynamodb", region_name=REGION)
    dynamodb.delete_table(TableName=table_name)
    logger.info(f"  Deleting {table_name}...")


def table_exists(table_name: str) -> bool:
    dynamodb = boto3.client("dynamodb", region_name=REGION)
    try:
        dynamodb.describe_table(TableName=table_name)
        return True
    except dynamodb.exceptions.ResourceNotFoundException:
        return False


def cleanup_s3(s3_prefix: str) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)
    for obj in resp.get("Contents", []):
        s3.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
    logger.info(f"  Cleaned up s3://{S3_BUCKET}/{s3_prefix}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Ingest clean.parquet → DynamoDB via S3 ImportTable")
    parser.add_argument("--parquet", type=Path, default=Path("builds/clean.parquet"))
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N rows (for testing)")
    parser.add_argument("--staging-only", action="store_true",
                        help="Import to staging table + verify only, don't replace production")
    args = parser.parse_args()

    if not args.parquet.exists():
        logger.error(f"Not found: {args.parquet}")
        sys.exit(1)

    s3_prefix = f"{S3_PREFIX}/current/"
    s3_key = f"{s3_prefix}variants.json.gz"
    local_gz = Path("/tmp/dynamo-import.json.gz")

    try:
        # 1. Convert
        logger.info("Step 1: Converting parquet → DynamoDB JSON")
        row_count = convert_parquet(args.parquet, local_gz, args.limit)

        # 2. Upload
        logger.info("Step 2: Uploading to S3")
        upload_to_s3(local_gz, s3_key)

        # 3. Import staging table
        if table_exists(STAGING_TABLE):
            logger.info(f"Cleaning up leftover staging table: {STAGING_TABLE}")
            delete_table(STAGING_TABLE)
            wait_for_delete(STAGING_TABLE)

        logger.info(f"Step 3: ImportTable → {STAGING_TABLE}")
        import_arn = start_import(STAGING_TABLE, s3_prefix)
        desc = wait_for_import(import_arn)
        errors = desc.get("ErrorCount", 0)
        if errors > 0:
            logger.error(f"{errors} import errors — check CloudWatch logs")
            sys.exit(1)

        # 4. Verify staging
        logger.info("Step 4: Verifying staging table")
        verify_table(STAGING_TABLE, row_count)
        logger.info("  Staging table verified OK")

        if args.staging_only:
            logger.info(f"Done (--staging-only). Staging table: {STAGING_TABLE}")
            return

        # 5. Delete old production table
        logger.info(f"Step 5: Deleting old table ({TABLE_NAME})")
        if table_exists(TABLE_NAME):
            delete_table(TABLE_NAME)
            wait_for_delete(TABLE_NAME)

        # 6. Import to production table name (DOWNTIME starts here)
        logger.info(f"Step 6: ImportTable → {TABLE_NAME} (downtime until complete)")
        import_arn = start_import(TABLE_NAME, s3_prefix)
        desc = wait_for_import(import_arn)
        errors = desc.get("ErrorCount", 0)
        if errors > 0:
            logger.error(f"{errors} import errors on production table — check CloudWatch")
            logger.error(f"  Staging table {STAGING_TABLE} still available as fallback")
            sys.exit(1)

        # 7. Verify production
        logger.info("Step 7: Verifying production table")
        verify_table(TABLE_NAME, row_count)

        # 8. Cleanup
        logger.info("Step 8: Cleanup")
        delete_table(STAGING_TABLE)
        cleanup_s3(s3_prefix)

        logger.info(f"Done. {TABLE_NAME} refreshed with {row_count:,} variants.")

    finally:
        if local_gz.exists():
            local_gz.unlink()


if __name__ == "__main__":
    main()
