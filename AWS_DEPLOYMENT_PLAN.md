# AWS Deployment Plan: Frontend Serving + Basic Data Retrieval

Goal: Serve the variant-viewer frontend from S3/CloudFront and back it with DynamoDB + API Gateway + Lambda for variant lookup and search. Lazy processing (SQS/Worker Lambda/analysis) is out of scope for now.

Reference: `/mnt/polished-lake/home/nnguyen/andromeda-scripts/genomics-app-architecture.md`

---

## Existing Assets

| Asset | Location | Notes |
|-------|----------|-------|
| Frontend SPA | `index.html` (this repo) | Single-file vanilla JS, no build step |
| App reference data | `global.json` (~2.4MB) | Head metadata, distributions, eval metrics |
| Pre-built variant JSONs | NFS: `.../mendelian/probe_webapp_build/variants/` | 232,789 files, ~18KB each |
| Search index | `search.json` (~17MB) | Gene-keyed, shape `{v, l, s, c}` per entry |

---

## Step Order

### Step 1: Scaffold directory structure

Create the repo layout for infra and Lambda code:

```
terraform/
  main.tf
  variables.tf
  outputs.tf
  modules/
    s3/
    cloudfront/
    dynamodb/
    api_gateway/
    lambda_api/
lambdas/
  api/
    handler.py
    requirements.txt
scripts/
  ingest.py
```

No logic yet — just the skeleton so subsequent steps have a place to land.

---

### Step 2: Terraform — S3 + CloudFront

Stand up the static frontend hosting:

- S3 bucket for `index.html` and `global.json` (private, CloudFront origin access)
- CloudFront distribution with:
  - Default behavior → S3 origin (static assets)
  - Long TTL for static assets
  - Custom error response: 404 → `/index.html` (SPA routing)

**Milestone:** Upload `index.html` + `global.json` to S3, access the app via CloudFront URL. Search and variant pages will be broken (no API yet), but the landing page should load.

---

### Step 3: Terraform — DynamoDB table + GSIs

Create the variants table:

- Partition key: `variant_id` (String)
- GSI-1: `variant_name` (partition) + `variant_id` (sort) — search by name
- GSI-2: `gene_number` (partition) + `variant_id` (sort) — search by gene
- Pay-per-request billing (scales to zero)

**Milestone:** Table exists, empty, ready for ingestion.

---

### Step 4: Bulk ingestion script

`scripts/ingest.py` — reads the 232K pre-built variant JSONs from NFS and batch-writes them to DynamoDB.

- Input: `/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian/probe_webapp_build/variants/`
- Parse each JSON, extract `variant_id`, `gene` (for GSI), and store the full record
- Use `batch_write_item` (25 items/call), parallelized
- Set `processing_status = not_started` on each item (for future lazy processing)

**Milestone:** 232K records in DynamoDB, queryable by variant_id, variant_name, and gene_number.

---

### Step 5: API Lambda — variant lookup + search

`lambdas/api/handler.py` — single Lambda handling two routes:

- `GET /variants/{id}` → DynamoDB `GetItem`, return variant JSON
- `GET /variants/search?q=...&field=variant_name|gene_number` → DynamoDB GSI `Query` with `begins_with`, return `[{v, l, s, c}, ...]`

The search response shape must match what `renderSearchResults()` in the frontend expects. The variant response shape must match the existing per-variant JSON structure.

---

### Step 6: Terraform — API Gateway + Lambda deployment

- HTTP API (API Gateway v2) with routes for `/variants/{id}` and `/variants/search`
- Lambda integration for the API handler
- CORS configuration for the CloudFront domain
- Add API Gateway as a second origin on the CloudFront distribution with `/variants/*` cache behavior

**Milestone:** API is live behind CloudFront. Can `curl` both endpoints and get correct responses.

---

### Step 7: Frontend changes

Modify `index.html` (phases 1–4 from the architecture doc's "Frontend Decoupling" section):

1. Add `CONFIG` object with `STATIC_BASE` and `API_BASE`
2. Prefix `global.json` fetch with `CONFIG.STATIC_BASE`
3. Replace client-side search (`search.json` + `clientSearch()`) with `apiSearch()` calling `GET /variants/search`
4. Replace per-variant static file fetch with `GET /variants/{id}` via `CONFIG.API_BASE`

**Milestone:** App works end-to-end via CloudFront — search returns results, variant pages load from DynamoDB.

---

### Step 8: Verify end-to-end

- [ ] CloudFront serves `index.html` and `global.json`
- [ ] Search by gene name returns results (e.g., "BRCA1")
- [ ] Clicking a search result loads the variant page from DynamoDB
- [ ] Direct navigation via URL hash works (`#/variant/{id}`)
- [ ] Landing page renders (UMAP optional — may not be deployed as static asset initially)
- [ ] CORS headers correct — no browser console errors
- [ ] API Gateway throttling configured (per-IP)
