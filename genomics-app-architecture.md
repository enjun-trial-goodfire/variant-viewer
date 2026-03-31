# Genomics Web App — Architecture

## Summary

A public-facing web app for researchers to browse ~4 million genomics records (~2000 columns each). Users search by variant name or gene number, view a record, and trigger lazy on-demand processing (local compute + LLM via OpenRouter/Claude). Processing results are stored after first computation — too costly to pre-compute for all 4M records.

**Stack:** AWS (serverless-first) · Terraform · Vite/React frontend
**Key constraint:** Low maintenance, small researcher audience, UX tolerance for async results (refresh to see output).

---

## Architecture Diagram

```
Browser
  │
  ├── static assets ──► CloudFront ──► S3 (frontend)
  │
  └── API calls ───────► CloudFront ──► API Gateway
                                          │
                                          ▼
                                     Lambda (API)  ──► DynamoDB
                                          │           (variants, job status,
                                          │            processed results)
                                          │
                                          └──► SQS FIFO queue
                                                    │
                                                    ▼
                                               Lambda (Worker)
                                                    ├── local compute
                                                    ├── OpenRouter API (Claude)
                                                    └── writes result ──► DynamoDB
                                                                          (or S3 if >400KB)
```

---

## Q&A: Co-worker Review Questions

### State in S3 — will we need to proxy large files via Lambda?

**No proxying needed.** The original plan considered S3 for storing full 2000-column records, with Lambda potentially reading and re-serving them. This is unnecessary. If records were in S3, Lambda would generate a presigned URL and return it to the frontend — the browser then fetches directly from S3. Lambda never touches the bytes.

However, the revised architecture (below) eliminates S3 for record storage entirely by using DynamoDB, so this is moot. Lambda returns DynamoDB records directly in its JSON response. Large processed results (if any exceed 400KB) use presigned S3 URLs, with the browser fetching directly.

---

### Does DynamoDB or a MongoDB-compatible DB avoid the Aurora + S3 bifurcation?

**Yes — DynamoDB eliminates the split entirely.** The original plan bifurcated storage into Aurora (for search indexing) and S3 (for full records) because PostgreSQL has a ~1600-column limit. DynamoDB avoids both problems:

- **No column limit** — stores full 2000-column records as document attributes
- **Item size limit is 400KB** — a 2000-column record of mostly numbers is ~40–80KB (attribute name + value ≈ 30 bytes × 2000), comfortably under the limit
- **Global Secondary Indexes (GSIs)** on `variant_name` and `gene_number` handle search
- **No connection pooling** — DynamoDB uses HTTP, not sockets; no RDS Proxy needed, no VPC required
- **Scales to zero** — no Aurora cold-start penalty (Aurora Serverless can take 5–30s to wake from idle)

**Recommended change: replace Aurora + RDS Proxy + S3-for-records with a single DynamoDB table.**

MongoDB-compatible (DocumentDB) would also unify the datastore, but it is not serverless — it requires always-on instances and adds maintenance overhead with no meaningful advantage here.

> **One open question:** DynamoDB GSIs support exact match and `begins_with` prefix search, but not substring or fuzzy search. If users need to type "BRCA" and see "BRCA1", "BRCA2" etc., a GSI sort key with `begins_with` works. If they need substring matching (e.g. "CA1" matching "BRCA1"), you'd need OpenSearch — which adds cost and maintenance. Clarify the search UX before committing.

---

### Does setting max concurrency on the worker Lambda prevent abusive usage?

**Partially — and it needs to be combined with two other layers.**

`ReservedConcurrentExecutions` on the worker Lambda caps how many jobs run simultaneously (protecting OpenRouter rate limits and your compute budget), but the SQS queue can still accumulate unboundedly if someone submits requests for thousands of different variants in a loop.

Three layers together handle this cleanly:

| Layer | What it prevents |
|-------|-----------------|
| **DB-level atomic deduplication** | Same variant cannot be queued/processed twice concurrently |
| **SQS FIFO deduplication** (MessageDeduplicationId = variant_id) | Belt-and-suspenders against race conditions |
| **API Gateway usage plan** (per-IP throttling) | Prevents bulk job submission across many different variants |

For a small researcher audience, per-IP throttling (e.g., 5 req/s, 50k req/day on the `/analysis` endpoint) combined with reserved concurrency of 10 on the worker is sufficient.

---

### Does Lambda need to proxy the full 2000-column records, or can the frontend get them directly from S3?

**Lambda does not proxy large payloads.** With the revised DynamoDB-based architecture, Lambda reads a record from DynamoDB and returns it directly in the HTTP response — no S3 fetch involved. DynamoDB reads are single-digit milliseconds.

If processed results from the LLM step are large (unlikely, but possible), those would be stored in S3 and Lambda would return a presigned URL. The browser fetches the file directly from S3 — Lambda is not in that data path.

---

### If Lambda proxy were necessary, how would that affect CloudFront caching?

Presigned S3 URLs bypass CloudFront for private buckets, so those fetches aren't cached at the edge — that's fine, since each URL is per-variant and short-lived anyway.

For the API itself, CloudFront in front of API Gateway enables caching on stable responses:

| Route | Cache behavior |
|-------|---------------|
| Static frontend assets | Long TTL (Vite outputs content-hashed filenames) |
| `GET /variants/{id}` (metadata) | Medium TTL — records rarely change |
| `GET /variants/search` | Short TTL or no cache — query-dependent |
| `GET /variants/{id}/analysis` | No cache — status changes over time |

Configure per-route cache policies on the CloudFront distribution, not a blanket TTL.

---

### How many Lambdas do we need? Is search backed by a Lambda? Is GET /variants/ backed by a Lambda?

**Minimum: 2 Lambdas.**

| Lambda | Trigger | Handles |
|--------|---------|---------|
| **API Lambda** | API Gateway | All HTTP routes: search, GET /variants/{id}, GET /variants/{id}/analysis |
| **Worker Lambda** | SQS FIFO | Async processing jobs (compute + OpenRouter call) |

One API Lambda handling all routes (via a simple router) is the right starting point. Splitting into per-route functions adds operational overhead with no benefit at this traffic level. Both search and record fetches go through the same API Lambda backed by DynamoDB.

---

### How do we ensure non-laggy UX with Lambda cold starts?

Lambda cold starts for a Python or Node.js function talking to DynamoDB (HTTP-based, no socket warmup) are typically **50–200ms** — meaningfully better than the original Aurora-based design, which required both Aurora's own wake-up latency and RDS Proxy connection establishment.

For a small researcher audience this may be acceptable as-is. Two clean options to eliminate cold starts on the API Lambda:

| Option | Monthly cost | Notes |
|--------|-------------|-------|
| **Provisioned Concurrency (1–2 instances)** on API Lambda | ~$15–30 | Eliminates cold starts; stays fully serverless |
| **ECS Fargate (0.25 vCPU / 0.5GB RAM)** for the API service | ~$10–15 | Always warm; slightly more operational surface but familiar to container-fluent teams |

The **worker Lambda** does not need warm instances — a cold start on an async job that takes minutes is irrelevant to UX.

---

### Meta: Is Lambda the right choice here?

**Lambda is a good fit, especially with DynamoDB replacing Aurora.** The original plan had a structural problem: Aurora (a socket-based relational DB) and Lambda are a poor pairing — Lambda needs RDS Proxy to avoid exhausting DB connections, which adds latency, cost, and a VPC requirement. DynamoDB's HTTP API eliminates this entirely.

With DynamoDB:
- No VPC required
- No RDS Proxy
- No connection pool management
- DynamoDB reads are single-digit milliseconds — faster than PostgreSQL at this scale
- Provisioned concurrency on the API Lambda makes simple GETs and pageloads consistently fast

**The only remaining Lambda concern is search latency** — if prefix search becomes a full-text search requirement, adding OpenSearch would argue for a persistent API service (ECS Fargate) over Lambda to avoid cold start costs there. But for exact/prefix match via DynamoDB GSI, Lambda is appropriate.

---

## Full Revised Architecture

### Data Storage: Single DynamoDB Table

**Primary table: `variant-viewer-variants`**

```
Partition key:  variant_id (String)    — e.g. "chr10:100042514:C:T"
```

**GSI: `gene-index`** — used by `GET /variants/search` for gene name lookup

```
Partition key:  gene (String)          — e.g. "BRCA1"
Sort key:       variant_id (String)
Projection:     INCLUDE (label, score, consequence)
```

The GSI projects only the fields needed for search results (`{v, l, s, c}` shape in the frontend). This keeps index storage small — the full variant record is only read via `GetItem` on the primary key. If search results need additional fields in the future, add them to the GSI projection.

> **Note:** The original design had two GSIs (one for variant_name, one for gene_number). In the actual data, variant IDs (e.g., `chr10:100042514:C:T`) are the primary key and users search by gene name, so one GSI on `gene` is sufficient. Direct variant lookup uses the primary key.

**Attributes per item:**

The table is schemaless beyond the keys. Each item stores the full variant JSON from `build.py` as top-level attributes (id, gene, score, label, consequence, disruption, effect, neighbors, etc.). New fields can be added without migration.

| Attribute | Type | Notes |
|-----------|------|-------|
| `variant_id` | String | Primary key — e.g. `chr10:100042514:C:T` |
| `gene` | String | GSI partition key — e.g. `BRCA1` |
| `label` | String | `benign` \| `pathogenic` \| `VUS` (projected in GSI) |
| `score` | Number | Pathogenicity score 0–1 (projected in GSI) |
| `consequence` | String | VEP consequence type (projected in GSI) |
| `processing_status` | String | `not_started` \| `pending` \| `processing` \| `complete` \| `failed` |
| `processed_at` | String | ISO timestamp, set on completion |
| `processed_result` | Map | LLM output (set on completion; use S3 + `result_s3_key` if >400KB) |
| `result_s3_key` | String | Optional — only if result exceeds DynamoDB item limit |
| *(all other fields)* | Various | Full variant record (~18KB): disruption, effect, neighbors, VEP annotations, gnomAD, ClinVar metadata |

DynamoDB pay-per-request billing scales to zero when idle.

**S3 Buckets (minimal):**

| Bucket | Purpose |
|--------|---------|
| `{app}-frontend` | Static assets served via CloudFront |
| `{app}-results` | Overflow bucket for processed results exceeding 400KB (rare) |

---

### Lazy Processing: Get-or-Create Flow

```
GET /variants/{id}/analysis

1. Read item from DynamoDB
   ├── processing_status = complete   → 200 + processed_result (or presigned URL if in S3)
   ├── processing_status = pending /
   │   processing                     → 202 + { status: "processing", message: "Refresh in ~60s" }
   └── processing_status = not_started / failed →

         Atomic conditional update:
           UpdateItem WHERE processing_status IN (not_started, failed)
           SET processing_status = pending

         If update succeeded (condition met):
           → Send message to SQS FIFO (MessageDeduplicationId = variant_id)
           → 202 + { status: "queued" }

         If update failed (condition not met — concurrent request already queued it):
           → 202 + { status: "processing" }
```

Frontend: on 202, display "Processing — please refresh in about a minute." Optionally auto-poll once after 60 seconds.

---

### Deduplication: Three Layers

| Layer | Mechanism | What it catches |
|-------|-----------|-----------------|
| DynamoDB conditional update | `WHERE status IN (not_started, failed)` | Primary gate — prevents double-enqueue |
| SQS FIFO deduplication | `MessageDeduplicationId = variant_id` (5-min window) | Race conditions that slip past layer 1 |
| Worker Lambda idempotency | Conditional update `WHERE status = pending` before processing; skip if 0 rows updated | Duplicate SQS message delivery |

---

### Worker Lambda

```
Trigger: SQS FIFO, batch size = 1

1. Atomic claim:
   UpdateItem SET status = processing WHERE status = pending
   └── 0 rows updated → duplicate delivery, return early

2. Read full record from DynamoDB (record_data attribute)

3. Run local computation (annotation, scoring, etc.)

4. Call OpenRouter API (Claude or similar)
   └── API key read from AWS Secrets Manager at cold start

5. Store result:
   └── If result < 400KB → write to DynamoDB processed_result attribute
   └── If result ≥ 400KB → write to S3, store key in result_s3_key attribute

6. UpdateItem SET status = complete, processed_at = now(), processed_result / result_s3_key

7. On any unrecoverable error:
   UpdateItem SET status = failed
   Log to CloudWatch
```

Config:
- **Timeout:** 14 min (just under Lambda's 15-min limit)
- **Memory:** 512MB–2GB (tune per workload)
- **Reserved concurrency:** 10 (caps parallel OpenRouter calls)
- **SQS visibility timeout:** > Lambda timeout (prevents premature requeue)
- **Dead letter queue:** Messages failing 3× go to SQS DLQ; CloudWatch alarm fires

---

### API Lambda

All HTTP routes handled by one Lambda function:

| Method | Path | DynamoDB operation |
|--------|------|--------------------|
| `GET` | `/variants/search?q=…&field=variant_name\|gene_number` | GSI query |
| `GET` | `/variants/{id}` | GetItem |
| `GET` | `/variants/{id}/analysis` | Conditional UpdateItem + SQS send (get-or-create) |

Config:
- **Provisioned concurrency:** 1–2 instances (eliminates cold starts on search and record fetch)
- **Timeout:** 10s (fast reads only; processing is async)
- **Memory:** 256MB

---

### Frontend

- **S3 + CloudFront** for static hosting (HTML/JS/CSS from Vite build)
- **CloudFront also in front of API Gateway** with per-route cache policies
- SPA (React/Vite or similar) with two main views:
  - **Search page** — query by variant name or gene number
  - **Record view** — display full record + analysis status; "Refresh" or auto-poll on `processing` state

---

### Bulk Data Ingestion (One-Time)

1. Obtain raw genomics data file (CSV, VCF, etc.)
2. Run a one-time Python ingestion script:
   - Parse each record
   - Batch write to DynamoDB via `batch_write_item` (25 items per call)
   - Set `processing_status = not_started` on all items
3. Run from a local machine or a small EC2 instance — no ongoing pipeline needed

At 4M records, with DynamoDB batch writes at ~25 items/call and a parallelized script, full ingestion completes in a few hours.

---

### Terraform Module Structure

```
terraform/
├── main.tf              # AWS provider, S3 backend for Terraform state
├── variables.tf         # app name, environment, region, OpenRouter secret ARN
├── outputs.tf           # CloudFront URL, API Gateway URL, DynamoDB table name
└── modules/
    ├── s3/              # frontend bucket, results overflow bucket
    ├── cloudfront/      # CDN for frontend + API Gateway origin
    ├── dynamodb/        # variants table + GSIs + auto-scaling policy
    ├── api_gateway/     # HTTP API, routes, CORS, usage plan (throttling)
    ├── lambda_api/      # API handler Lambda, provisioned concurrency, IAM role
    ├── lambda_worker/   # Processing worker Lambda, reserved concurrency, IAM role
    ├── sqs/             # FIFO queue, DLQ, CloudWatch alarm on DLQ depth
    └── secrets/         # OpenRouter API key in Secrets Manager
```

No VPC module needed — DynamoDB and SQS are AWS-managed service endpoints; Lambdas reach them without VPC configuration.

---

## Open Questions

| # | Question | Recommendation |
|---|----------|----------------|
| 1 | **Search type** | Exact match or `begins_with` prefix → DynamoDB GSI is sufficient. Substring/fuzzy → add OpenSearch Serverless (increases maintenance). Clarify before building. |
| 2 | **Auth** | Start unauthenticated with API Gateway per-IP throttling. Upgrade path: Cognito email sign-up in front of the `/analysis` endpoint if abuse occurs. |
| 3 | **Result expiry** | Should processed results ever be invalidated (e.g., if underlying data or the LLM prompt changes)? If yes, add a `result_version` attribute and a one-time re-processing script. |
| 4 | **Record size validation** | Before committing to DynamoDB, validate a sample of real records to confirm they fit under 400KB. Edge cases (unusually dense records) may need the S3 overflow path. |
| 5 | **CI/CD: auto-deploy frontend** | Set up a GitHub Actions workflow that deploys `index.html` to the S3 frontend bucket (`variant-viewer-frontend`) when it changes on `master`. Should also invalidate the CloudFront cache so changes go live immediately. Repo has no existing CI/CD. Workflow needs AWS credentials (via OIDC or repo secrets) and the CloudFront distribution ID. |
| 6 | **Search prefix scaling at 4M records** | See "Search Prefix Scaling" section below for details. The current gene search uses a GSI exact match + scan-with-filter fallback for prefix matching. This works at 232K records but will degrade at 4M. Needs a prefix-friendly index design or OpenSearch before production scale. |

---

## Search Prefix Scaling

### The problem

Users search by typing a gene name prefix (e.g., "BRC" to find "BRCA1", "BRCA2"). The current implementation in `lambdas/api/handler.py` handles this in two steps:

1. **Exact match via GSI query**: `Key("gene").eq("BRC")` — fast, indexed, but only finds genes named exactly "BRC" (none exist).
2. **Prefix fallback via scan with filter**: `FilterExpression="begins_with(gene, :prefix)"` scanning up to 1000 rows from the `gene-index` GSI.

The scan fallback is the issue. DynamoDB scans read rows sequentially and discard non-matches. At 232K records this is fast (the scan finishes quickly). At 4M records, the scan may need to read many more rows before finding matches, and DynamoDB charges per row **scanned**, not per row returned. Latency and cost both increase.

### Why the GSI can't do prefix search natively

The `gene-index` GSI uses `gene` as its **partition key**. DynamoDB can only do exact equality on partition keys — there's no `begins_with` on a partition key. The `begins_with` operator only works on **sort keys** within a single partition.

### Solution A: Prefix-friendly GSI (recommended for this use case)

Add a new GSI designed for prefix search:

```
GSI: gene-prefix-index
  Partition key:  gene_prefix (String)    — first 3 characters of gene name, e.g. "BRC"
  Sort key:       gene (String)           — full gene name, e.g. "BRCA1"
  Projection:     INCLUDE (variant_id, label, score, consequence)
```

At ingestion time, compute `gene_prefix = gene[:3].upper()` and store it as an attribute. The search handler then:

1. Compute `prefix = query[:3].upper()`
2. Query the GSI: `Key("gene_prefix").eq(prefix) & Key("gene").begins_with(query)`
3. Fully indexed — no scan, O(results) not O(table)

**Trade-off**: Adds ~1 byte per item storage overhead. Genes sharing the same 3-letter prefix land in the same partition, but gene name cardinality is low (~20K unique genes) so partition sizes stay small.

**Implementation**: Requires changes to:
- `terraform/modules/dynamodb/main.tf` — add the new GSI and `gene_prefix` attribute
- `scripts/ingest.py` — compute and store `gene_prefix` during ingestion
- `lambdas/api/handler.py` — replace the scan fallback with a GSI query

### Solution B: OpenSearch Serverless

For substring search ("CA1" matching "BRCA1"), fuzzy search, or multi-field search (gene + consequence + disease), add OpenSearch Serverless as a search index alongside DynamoDB.

**Trade-off**: Adds operational complexity (~$25/month minimum for serverless), a sync mechanism (DynamoDB Streams → Lambda → OpenSearch), and a second data store to keep consistent. Only justified if search requirements go beyond prefix matching.

### Current status

The scan fallback is sufficient for the 232K test dataset. Revisit before scaling to 4M records. Solution A is the recommended next step.

---

## Verification Checklist

- [ ] `terraform plan` runs clean in a fresh AWS account with correct resource names
- [ ] Bulk ingestion script populates DynamoDB for a sample of 1,000 records within expected time
- [ ] Search by variant_name and gene_number returns correct records via GSI
- [ ] First `GET /variants/{id}/analysis` enqueues job, returns 202
- [ ] Concurrent requests to same variant_id return 202 without double-enqueue (check SQS message count = 1)
- [ ] Worker Lambda processes job, writes result to DynamoDB, sets `status = complete`
- [ ] Second `GET /variants/{id}/analysis` returns 200 with result
- [ ] Worker Lambda DLQ CloudWatch alarm fires on repeated failures
- [ ] API Lambda provisioned concurrency keeps p99 latency under 200ms for search and record fetch
- [ ] CloudFront serves frontend; API Gateway CORS configured for frontend domain
- [ ] Per-IP throttling on API Gateway rejects excessive `/analysis` calls

---

## Frontend Decoupling: Static JSON → API Backend

### Background

The frontend lives at `/mnt/polished-lake/home/nnguyen/variant-viewer/index.html` — a single-file vanilla JS SPA (no framework, no build tool, all JS inline). It currently fetches data from static JSON files generated by `build.py`:

| Fetch | Current URL | Trigger | Size |
|-------|------------|---------|------|
| Head metadata, distributions, eval | `global.json` (relative) | Page load, blocking | ~300KB |
| UMAP embedding | `umap.json` (relative) | Landing page shown | ~2.3MB |
| Search index | `search.json` (relative) | First search input | ~17MB |
| Per-variant record | `variants/{safeVariantId(id)}.json` (relative) | Variant page navigation | ~18KB each |
| Claude interpretation | `/api/interpret/{id}` (absolute) | After variant renders | On-demand |

`safeVariantId()` (lines 237–244) replaces `:` and `/` with `_` and hashes IDs >200 chars — needed for filesystem filenames, unnecessary with DynamoDB.

There is **no configurable base URL** — all paths are hardcoded. There is **no polling** — the interpretation call is fire-and-forget with silent failure.

### Goal

Modify `index.html` so the frontend:
1. Serves as a pure static asset from S3 + CloudFront (HTML/CSS/JS only, no data files)
2. Fetches variant data and search results from API Gateway → Lambda → DynamoDB
3. Handles the 202 async polling pattern for on-demand analysis
4. Keeps `global.json` and `umap.json` as static assets (they are dataset-level reference data, not per-variant records)

---

### Phase 1: Add API config object

Insert at **line 216** (before the `// ── State` section):

```js
const CONFIG = Object.assign({
  STATIC_BASE: '',   // static assets (global.json, umap.json) — same origin in prod
  API_BASE: '',      // API Gateway origin — same CloudFront distribution in prod
}, window.__APP_CONFIG__ || {});
```

Deployers override via a `<script>` tag before the main script: `window.__APP_CONFIG__ = { API_BASE: 'https://...' }`. No build tool needed. With defaults (`''`), all URLs remain relative (backwards-compatible with `serve.py`).

---

### Phase 2: Prefix static asset fetches

`global.json` and `umap.json` are pre-computed dataset-level data (head schemas, distributions, UMAP coordinates). They stay as static files in S3 with long CloudFront cache TTLs.

| Line | Current | New |
|------|---------|-----|
| 299 | `fetch('global.json')` | `` fetch(`${CONFIG.STATIC_BASE}/global.json`) `` |
| 306 | `fetch('umap.json')` | `` fetch(`${CONFIG.STATIC_BASE}/umap.json`) `` |

No behavior change — purely wiring through the config.

---

### Phase 3: Replace client-side search with server-side API

The 17MB `search.json` client-side index cannot scale to 4M records.

**Remove** (dead code after this change):
- `searchIndex` and `searchIndexPromise` variables (line 223)
- `ensureSearchIndex()` function (lines 310–313)
- `clientSearch()` function (lines 327–332)

**Add** `apiSearch()` in the Search section (~line 327):

```js
async function apiSearch(query) {
  const q = query.trim();
  if (q.length < 2) return [];
  const field = q.includes(':') ? 'variant_name' : 'gene_number';
  const resp = await fetch(`${CONFIG.API_BASE}/variants/search?q=${encodeURIComponent(q)}&field=${field}`);
  if (!resp.ok) return [];
  return resp.json();
}
```

The API must return the same shape the frontend already consumes: `[{v, l, s, c}, ...]` (variant ID, label, score, consequence). This means `renderSearchResults()` (lines 334–341) requires **zero changes**.

**Update search input handler** (line 352):
```js
// was: setTimeout(() => ensureSearchIndex().then(() => renderSearchResults(clientSearch(q))), 150);
searchTimeout = setTimeout(() => apiSearch(q).then(r => renderSearchResults(r)), 300);
```

**Update Enter key handler** (line 359):
```js
// was: ensureSearchIndex().then(() => { const r = clientSearch(q); if (r.length) navigate(`variant/${r[0].v}`); });
apiSearch(q).then(r => { if (r.length) navigate(`variant/${r[0].v}`); });
```

Debounce bumped from 150ms → 300ms to account for network round-trip. Direct variant ID navigation (lines 348–350: input contains `:` with 4+ parts) stays unchanged.

---

### Phase 4: Change per-variant fetch to API endpoint

**Line 603** — replace static file fetch with API call:

```js
// was: fetch(`variants/${safeVariantId(variantId)}.json`)
const variantReady = fetch(`${CONFIG.API_BASE}/variants/${encodeURIComponent(variantId)}`)
  .then(r => { if (!r.ok) throw new Error(); return r.json(); });
```

`safeVariantId()` (lines 237–244) becomes dead code — remove it. DynamoDB uses the real variant ID as partition key; `encodeURIComponent()` handles URL-safety.

The rest of `loadVariant()` (Promise.all with globalReady, error display) stays the same.

---

### Phase 5: Implement 202 polling for analysis

Replace `fetchInterpretation()` (lines 615–632) and its call site (line 855) with a polling-aware version that handles the get-or-create pattern from the architecture doc.

**Replace lines 615–632 with:**

```js
let analysisAbort = null;

async function fetchAnalysis(v) {
  analysisAbort?.abort();
  analysisAbort = new AbortController();
  const signal = analysisAbort.signal;
  const container = document.getElementById('interp-container');
  if (!container) return;
  container.style.display = '';
  container.innerHTML = `<div class="interp-loading"><div class="interp-spinner"></div>
    <span>Checking for analysis...</span></div>`;
  try {
    const result = await pollAnalysis(v.id, signal);
    if (result) renderAnalysis(container, result);
    else container.style.display = 'none';
  } catch { container.style.display = 'none'; }
}

async function pollAnalysis(variantId, signal, maxAttempts = 10) {
  const url = `${CONFIG.API_BASE}/variants/${encodeURIComponent(variantId)}/analysis`;
  for (let i = 0; i < maxAttempts; i++) {
    const resp = await fetch(url, { signal });
    if (resp.status === 200) return resp.json();
    if (resp.status === 202) {
      const body = await resp.json();
      const container = document.getElementById('interp-container');
      if (container) container.querySelector('span').textContent = body.message || 'Processing analysis...';
      await new Promise(r => setTimeout(r, Math.min(30000, 2000 * 2 ** i)));
      continue;
    }
    return null;
  }
  return null;
}

function renderAnalysis(container, interp) {
  if (!interp || interp.status !== 'ok') { container.style.display = 'none'; return; }
  const cc = interp.confidence === 'high' ? 'conf-high'
           : interp.confidence === 'medium' ? 'conf-medium' : 'conf-low';
  container.innerHTML = `
    <div class="section-title">Variant Interpretation <span class="interp-confidence ${cc}">${interp.confidence} confidence</span></div>
    <div class="interp-summary">${interp.summary}</div>
    <div class="interp-mechanism"><b>Mechanism:</b> ${interp.mechanism}</div>
    <div class="section-title" style="margin-top:12px;font-size:11px">Key Evidence</div>
    <ul class="interp-evidence">${interp.key_evidence.map(e => '<li>' + e + '</li>').join('')}</ul>
    <div style="font-size:10px;color:var(--text-muted);margin-top:8px">Generated by ${interp.model || 'Claude'}</div>`;
  container.style.display = '';
}
```

**Line 855:** Change `fetchInterpretation(v)` → `fetchAnalysis(v)`

Polling uses exponential backoff (2s, 4s, 8s, ... capped at 30s). `AbortController` cancels in-flight polls when the user navigates away from a variant.

---

### Deploy Order

Phases are independent and can be deployed incrementally:

1. **Phase 1 + 2** — config plumbing + static asset prefixes. Zero behavior change, no backend needed.
2. **Phase 4** — per-variant API fetch. Requires `GET /variants/{id}` Lambda to be live.
3. **Phase 5** — analysis polling. Requires worker Lambda + SQS + `GET /variants/{id}/analysis`.
4. **Phase 3** — server-side search. Requires `GET /variants/search` Lambda + DynamoDB GSIs. Biggest UX change — do last.

### Frontend Decoupling Checklist

- [ ] `CONFIG` object added; `window.__APP_CONFIG__` override works
- [ ] `global.json` and `umap.json` fetched via `STATIC_BASE` prefix
- [ ] `search.json` no longer fetched; `apiSearch()` calls `GET /variants/search`
- [ ] Per-variant fetch uses `GET /variants/{id}` with `encodeURIComponent`; `safeVariantId()` removed
- [ ] Analysis endpoint returns 200 → renders immediately; returns 202 → polls with backoff
- [ ] Navigating away from a variant cancels in-flight analysis poll
- [ ] `renderSearchResults()` works unchanged (API returns `{v, l, s, c}` shape)
- [ ] App still works against local `serve.py` with default empty config

---

## Current Deployment Status (as of 2026-03-31)

### What's deployed and working

| Component | Status | Details |
|-----------|--------|---------|
| **S3 + CloudFront** | Live | `variant-viewer-frontend` bucket, CloudFront distribution `E2MJCHKBJN4TYI`, URL: `https://d3v7t6hzw7vnvv.cloudfront.net` |
| **DynamoDB** | Live, populated | `variant-viewer-variants` table, 232,766 records ingested, GSI `gene-index` on `gene` field |
| **API Gateway** | Live | `https://xix0d0o8le.execute-api.us-east-1.amazonaws.com/`, routes: `GET /variants/{proxy+}`, `GET /variants/search` |
| **API Lambda** | Live | `variant-viewer-api`, Python 3.12, reads from DynamoDB, includes disruption format normalization |
| **Frontend** | Live, dual-mode | `index.html` with CONFIG object, `config.js` in S3 sets `API_BASE` for deployed mode |
| **Terraform state** | S3 backend | Bucket: `variant-viewer-terraform-state`, key: `variant-viewer/terraform.tfstate` |

### What's NOT deployed yet

| Component | Notes |
|-----------|-------|
| **Worker Lambda + SQS** | Lazy on-demand analysis (LLM interpretation). Planned in architecture but out of scope for initial deployment. |
| **Analysis polling (frontend Phase 5)** | Frontend code for 202 polling not implemented yet — depends on Worker Lambda. |
| **CI/CD (GitHub Actions)** | Auto-deploy `index.html` to S3 on push to `master`. Documented in Open Questions #5. |
| **Per-IP throttling** | API Gateway usage plan not configured yet. |
| **CloudFront API origin** | API calls currently go directly to API Gateway. Adding API Gateway as a second CloudFront origin would unify the domain and enable caching. |

### Repos and key files

| Repo | Path | Branch | Purpose |
|------|------|--------|---------|
| **variant-viewer** | `/mnt/polished-lake/home/nnguyen/variant-viewer` | `aws-deployment` | Frontend app + Terraform + Lambda + ingestion script |
| **andromeda-scripts** | `/mnt/polished-lake/home/nnguyen/andromeda-scripts` | `main` | This architecture doc |

Key files in `variant-viewer` (branch `aws-deployment`):

```
index.html                          # Frontend SPA (dual-mode: static files or API)
config.js.example                   # Template for deployed config (sets API_BASE)
AWS_DEPLOYMENT_PLAN.md              # Step-by-step deployment plan
terraform/
  main.tf                           # Root module wiring S3, CloudFront, DynamoDB, Lambda, API Gateway
  modules/s3/                       # Private S3 bucket for static assets
  modules/cloudfront/               # CDN with OAC, SPA error handling (404→index.html)
  modules/dynamodb/                 # Variants table + gene-index GSI
  modules/lambda_api/               # API Lambda with IAM, auto-zipped from lambdas/api/
  modules/api_gateway/              # HTTP API v2, CORS, routes, Lambda integration
lambdas/api/handler.py              # Lambda handler: GET /variants/{id}, GET /variants/search
scripts/ingest.py                   # Bulk ingestion: variant JSONs → DynamoDB (supports --wipe)
```

### Known issues to address

1. **Disruption data format mismatch**: The ingested data (from `probe_webapp_build`) stores disruption values as `[ref, var]` pairs (older `build.py` format). The frontend expects delta scalars. A normalization shim in the Lambda converts on the fly. **Fix**: re-run `build.py` (latest version) and re-ingest with `python scripts/ingest.py <new_build_dir> --wipe`. The Lambda shim detects the format and skips conversion when deltas are already scalars.

2. **23 failed ingestion records**: Indel variants with very long allele sequences (up to 8,732 bytes in the variant ID) exceed DynamoDB's key size limits (2048 bytes for partition key, 1024 bytes for GSI sort key). These are structural variants like `chr11:20628016:TATCTGCTGCATGGGG...`. Root cause confirmed — the `batch_writer` error messages are misleading (attributed to nearby files, not the actual failing item). Options: hash long IDs with a `variant_id_full` attribute, or skip them (0.01% of dataset).

3. **Frontend `distributions` field name**: The `global.json` in S3 uses `distribution` (singular) as the key for the pathogenicity distribution data, while the frontend code references `distributions` (plural). This was carried over from the pre-existing build. Check whether the current `index.html` handles both, or if the field name in `global.json` needs updating when the data is rebuilt.

### Where step 8 (verification) left off

The Lambda disruption normalization fix was deployed but the variant detail page has not been re-tested in the browser since. The next session should:

1. **Hard refresh** `https://d3v7t6hzw7vnvv.cloudfront.net` and verify:
   - Landing page loads with UMAP visualization
   - Search for "BRCA1" returns results
   - Clicking a variant loads the detail page without console errors
   - Disruption profile bars render correctly (diverging bars with delta values)
   - Effect predictions section renders
   - Score distributions (Chart.js histograms) render
2. **Check the `distributions` key**: the frontend code uses `g.distributions` but `global.json` may use `distribution` (singular). If histograms don't render, this is likely why — fix by updating the key in `global.json` or adding a fallback in `index.html`.
3. **Test edge cases**: try a VUS variant, a benign variant, a variant with no neighbors.

### How to resume work

**Prerequisites**: AWS credentials configured, Terraform installed at `~/.local/bin/terraform`.

**To deploy Terraform changes**:
```bash
cd /mnt/polished-lake/home/nnguyen/variant-viewer/terraform
export PATH="$HOME/.local/bin:$PATH"
terraform init    # only needed after adding/changing modules
terraform plan    # preview
terraform apply   # deploy
```

**To re-ingest data** (after team re-runs `build.py` with latest version):
```bash
cd /mnt/polished-lake/home/nnguyen/variant-viewer
python3 scripts/ingest.py /path/to/new/build/variants/ --wipe
```

**To update the frontend on S3**:
```bash
aws s3 cp index.html s3://variant-viewer-frontend/index.html --content-type "text/html"
aws cloudfront create-invalidation --distribution-id E2MJCHKBJN4TYI --paths "/index.html"
```

**To update the Lambda code** (after editing `lambdas/api/handler.py`):
```bash
cd /mnt/polished-lake/home/nnguyen/variant-viewer/terraform
export PATH="$HOME/.local/bin:$PATH"
terraform apply   # detects code change via source_code_hash, redeploys Lambda
```

**Source data location** (NFS, requires sudo or group membership):
```
/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian/probe_webapp_build/
  variants/       # 232,789 per-variant JSON files (~18KB each)
  global.json     # Head metadata, distributions, eval metrics, UMAP data
  index.html      # Old version of frontend (pre-AWS changes)
```
