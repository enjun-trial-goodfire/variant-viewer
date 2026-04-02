#!/usr/bin/env bash
# Deploy EVEE to AWS. Run from the repo root.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Terraform installed
#   - builds/clean.parquet exists (run: uv run vv transform --probe probe_v12)
#   - builds/heads.json and builds/statistics.json exist (created by transform)
#   - frontend/dist/ exists (run: cd frontend && npm run build)
#
# Usage:
#   ./scripts/deploy.sh              # full deploy (infra + data + frontend)
#   ./scripts/deploy.sh --data-only  # skip terraform, just re-ingest + upload
#   ./scripts/deploy.sh --frontend-only  # just rebuild + upload frontend

set -euo pipefail
cd "$(dirname "$0")/.."

BUILDS=builds
FRONTEND=frontend/dist

# Parse flags
DATA_ONLY=false
FRONTEND_ONLY=false
for arg in "$@"; do
  case $arg in
    --data-only) DATA_ONLY=true ;;
    --frontend-only) FRONTEND_ONLY=true ;;
  esac
done

echo "=== EVEE Deploy ==="

# ── 1. Verify artifacts exist ────────────────────────────────────────
echo ""
echo "Checking artifacts..."
for f in "$BUILDS/clean.parquet" "$BUILDS/heads.json" "$BUILDS/statistics.json"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Missing $f — run: uv run vv transform --probe probe_v12"
    exit 1
  fi
done

if [ ! -d "$FRONTEND" ]; then
  echo "ERROR: Missing $FRONTEND — run: cd frontend && npm run build"
  exit 1
fi

echo "  All artifacts present."

# ── 2. Terraform (skip if --data-only or --frontend-only) ────────────
if [ "$DATA_ONLY" = false ] && [ "$FRONTEND_ONLY" = false ]; then
  echo ""
  echo "Applying Terraform..."
  cd terraform
  terraform init -input=false
  terraform apply -auto-approve
  cd ..
fi

# ── 3. Get resource names from Terraform ─────────────────────────────
echo ""
echo "Reading Terraform outputs..."
cd terraform
S3_BUCKET=$(terraform output -raw frontend_bucket_name 2>/dev/null || echo "")
CLOUDFRONT_ID=$(terraform output -raw cloudfront_distribution_id 2>/dev/null || echo "")
API_URL=$(terraform output -raw api_url 2>/dev/null || echo "")
cd ..

if [ -z "$S3_BUCKET" ]; then
  echo "WARNING: Could not read Terraform outputs. Using defaults."
  S3_BUCKET="variant-viewer-frontend"
fi

# ── 4. Ingest data to DynamoDB (skip if --frontend-only) ─────────────
if [ "$FRONTEND_ONLY" = false ]; then
  echo ""
  echo "Ingesting to DynamoDB (S3 ImportTable)..."
  uv run python scripts/ingest.py --parquet "$BUILDS/clean.parquet"
fi

# ── 5. Upload static assets to S3 ───────────────────────────────────
echo ""
echo "Uploading frontend to S3..."
aws s3 sync "$FRONTEND" "s3://$S3_BUCKET/" --delete

echo "Uploading data assets..."
aws s3 cp "$BUILDS/heads.json" "s3://$S3_BUCKET/heads.json"
aws s3 cp "$BUILDS/statistics.json" "s3://$S3_BUCKET/statistics.json"

# UMAP is optional (only if computed with --umap)
if [ -f "$BUILDS/umap.json" ]; then
  aws s3 cp "$BUILDS/umap.json" "s3://$S3_BUCKET/umap.json"
fi

# ── 6. Write config.js ──────────────────────────────────────────────
if [ -n "$API_URL" ]; then
  echo "Writing config.js with API_BASE=$API_URL"
  echo "window.__APP_CONFIG__ = { API_BASE: '$API_URL' };" > /tmp/config.js
  aws s3 cp /tmp/config.js "s3://$S3_BUCKET/config.js"
fi

# ── 7. Invalidate CloudFront cache ──────────────────────────────────
if [ -n "$CLOUDFRONT_ID" ]; then
  echo "Invalidating CloudFront cache..."
  aws cloudfront create-invalidation --distribution-id "$CLOUDFRONT_ID" --paths "/*" > /dev/null
fi

echo ""
echo "=== Deploy complete ==="
if [ -n "$API_URL" ]; then
  echo "Site: ${API_URL/\/api/}"
fi
