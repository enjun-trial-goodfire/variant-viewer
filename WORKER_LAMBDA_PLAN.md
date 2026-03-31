# Plan: Worker Lambda + SQS + Analysis Polling

## Context

The variant-viewer app is deployed on AWS with a working frontend (S3 + CloudFront), API Lambda (search + variant lookup), and DynamoDB (232K records). The missing piece is on-demand AI interpretation: when a user views a variant, we call Claude to generate a clinical interpretation. This requires a Worker Lambda triggered by SQS, a get-or-create endpoint in the API Lambda, and polling logic in the frontend. All 232K DynamoDB records already have `processing_status = "not_started"`.

## Prerequisites (manual, before terraform apply)

Create the Anthropic API key secret in AWS Secrets Manager:
```bash
aws secretsmanager create-secret \
  --name variant-viewer/anthropic-api-key \
  --secret-string "sk-ant-..." \
  --region us-east-1
```
Note the ARN for use as a Terraform variable.

---

## Step 1: Create SQS Terraform module

**New files:** `terraform/modules/sqs/{main,variables,outputs}.tf`

- FIFO queue `${app_name}-processing.fifo` with explicit `MessageDeduplicationId` (not content-based)
- DLQ `${app_name}-processing-dlq.fifo` with `maxReceiveCount = 3`
- `visibility_timeout_seconds = 900` (must exceed worker Lambda timeout)
- Outputs: `queue_url`, `queue_arn`, `dlq_arn`

## Step 2: Create Worker Lambda code

**New file:** `lambdas/worker/handler.py`

- **Zero external dependencies** — calls Anthropic Messages API via `urllib.request` (no SDK needed)
- Imports `prompts.py` (bundled in zip by Terraform, see Step 3)
- SQS handler processes one message at a time (batch_size=1)
- Flow:
  1. Parse `variant_id` from SQS message body
  2. Atomic claim: `UpdateItem SET processing_status = "processing" WHERE processing_status = "pending"` — returns early if already claimed (idempotency)
  3. Load full variant from DynamoDB, convert Decimals to floats, build prompt via `build_prompt(v)`
  4. POST to `https://api.anthropic.com/v1/messages` with JSON schema output format (same schema as `serve.py`)
  5. On success: write `processed_result` to DynamoDB, set `processing_status = "complete"`
  6. On failure: set `processing_status = "failed"`, re-raise for SQS retry
- API key fetched from Secrets Manager at cold start, cached in module-level global
- Reuses: `prompts.py:build_prompt()`, `prompts.py:SYSTEM_PROMPT`, `constants.py:calibration_text()`

## Step 3: Create Worker Lambda Terraform module

**New files:** `terraform/modules/lambda_worker/{main,variables,outputs}.tf`

- Function `${app_name}-worker`, Python 3.12, timeout 840s, memory 512MB
- `reserved_concurrent_executions = 10` (caps parallel Claude API calls)
- SQS event source mapping with `batch_size = 1`
- IAM policies: DynamoDB GetItem+UpdateItem, SQS Receive+Delete, Secrets Manager GetSecretValue, CloudWatch Logs
- Env vars: `TABLE_NAME`, `ANTHROPIC_SECRET_ARN`
- **Packaging**: Use `archive_file` with multiple `source` blocks (no file duplication):
  - `lambdas/worker/handler.py` → `handler.py`
  - `prompts.py` → `prompts.py`
  - `constants.py` → `constants.py`
  - `display.py` → `display.py`
  - `head_quality.json` → `head_quality.json`

  All files land at `/var/task/` in Lambda. `Path("head_quality.json")` in prompts.py resolves correctly.

## Step 4: Add analysis endpoint to API Lambda

**Modify:** `lambdas/api/handler.py`

Add `SQS_QUEUE_URL` env var (with graceful fallback if empty) and `boto3.client("sqs")`.

Add `handle_get_analysis(variant_id)` implementing the get-or-create pattern:
- `processing_status == "complete"` → **200** with `{status: "complete", result: {...}}`
- `processing_status in ("pending", "processing")` → **202** with `{status: "processing"}`
- `processing_status in ("not_started", "failed")` → atomic conditional update to `"pending"` + SQS SendMessage → **202** with `{status: "queued"}`
- Race condition on conditional update → **202** (safe fallback)

Update routing: check `path.endswith("/analysis")` **before** the existing `/variants/{id}` catch-all. Variant IDs never contain `/`, so this is safe.

SQS message: `MessageGroupId = variant_id` (allows parallelism across variants), `MessageDeduplicationId = variant_id` (5-min dedup window).

## Step 5: Update API Lambda Terraform module

**Modify:** `terraform/modules/lambda_api/{main,variables}.tf`

- Add variables: `sqs_queue_url` (default `""`), `sqs_queue_arn` (default `""`)
- Add IAM policy: `dynamodb:UpdateItem` on table ARN
- Add IAM policy (conditional on `sqs_queue_arn != ""`): `sqs:SendMessage` on queue ARN
- Add env var: `SQS_QUEUE_URL = var.sqs_queue_url`

## Step 6: Wire modules in root Terraform

**Modify:** `terraform/main.tf`
- Add `module "sqs"` block
- Add `module "lambda_worker"` block (depends on sqs, dynamodb)
- Update `module "lambda_api"` to pass `sqs_queue_url` and `sqs_queue_arn`

**Modify:** `terraform/variables.tf`
- Add `anthropic_secret_arn` variable (no default — required)

**Modify:** `terraform/outputs.tf`
- Add `sqs_queue_url` and `worker_function_name` outputs

## Step 7: Update frontend polling

**Modify:** `index.html`

Add `let analysisAbortController = null;` to state section (~line 240).

Replace `fetchInterpretation(v)` (lines 699-716) with:
- `fetchAnalysis(v)` — entry point, aborts previous controller, determines URL based on `CONFIG.API_BASE`
- `pollAnalysis()` — recursive polling with exponential backoff (2s base, 1.5x multiplier, 30s cap, 20 max attempts)
- `renderInterpretation(container, interp)` — extracted rendering (reuses existing HTML/CSS unchanged)

Dual-mode support:
- Deployed (`CONFIG.API_BASE` set): hits `GET /variants/{id}/analysis`, handles 200/202
- Local dev (`CONFIG.API_BASE` empty): hits `/api/interpret/{id}` (serve.py), gets 200 directly

AbortController cancels in-flight fetches + pending timeouts when navigating away.

Update call site at line 958: `fetchInterpretation(v)` → `fetchAnalysis(v)`

Existing CSS (`.interp-card`, `.interp-loading`, `.interp-spinner`, `.conf-*`) is reused unchanged.

---

## Deploy sequence

1. `terraform plan -var="anthropic_secret_arn=arn:aws:secretsmanager:..."` — expect ~11 new resources, 1 changed
2. `terraform apply` — creates SQS, Worker Lambda, updates API Lambda
3. Test API: `curl $API_URL/variants/chr17:43092919:G:A/analysis` — expect 202, then 200 after ~30s
4. Check worker logs: `aws logs tail /aws/lambda/variant-viewer-worker --since 5m`
5. Upload frontend: `aws s3 cp index.html s3://variant-viewer-frontend/index.html --content-type "text/html"`
6. Invalidate cache: `aws cloudfront create-invalidation --distribution-id E2MJCHKBJN4TYI --paths "/index.html"`
7. Browser test: search BRCA1 → click variant → see spinner → interpretation appears

## Files

**New (7):**
| File | Purpose |
|------|---------|
| `lambdas/worker/handler.py` | Worker Lambda: claim, prompt, call Claude, store result |
| `terraform/modules/sqs/main.tf` | FIFO queue + DLQ |
| `terraform/modules/sqs/variables.tf` | |
| `terraform/modules/sqs/outputs.tf` | |
| `terraform/modules/lambda_worker/main.tf` | Worker Lambda + IAM + SQS trigger |
| `terraform/modules/lambda_worker/variables.tf` | |
| `terraform/modules/lambda_worker/outputs.tf` | |

**Modified (7):**
| File | Change |
|------|--------|
| `lambdas/api/handler.py` | Add `handle_get_analysis()` route + SQS enqueue |
| `terraform/modules/lambda_api/main.tf` | Add UpdateItem + SendMessage IAM, add SQS_QUEUE_URL env |
| `terraform/modules/lambda_api/variables.tf` | Add `sqs_queue_url`, `sqs_queue_arn` |
| `terraform/main.tf` | Wire sqs + lambda_worker modules, pass SQS to lambda_api |
| `terraform/variables.tf` | Add `anthropic_secret_arn` |
| `terraform/outputs.tf` | Add SQS + worker outputs |
| `index.html` | Replace fetchInterpretation with polling fetchAnalysis |
