"""DuckDB-backed API server for the variant viewer.

The DB schema matches the frontend Variant type exactly (finalize_schema
runs at build time). The server is a thin query proxy — no transformation.

Usage:
    ANTHROPIC_API_KEY=... uv run vv serve [--db builds/variants.duckdb] [--port 8501]
"""

import asyncio
import json
import os
import time
from pathlib import Path

import duckdb
import orjson
from loguru import logger
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route

import anthropic

from prompts import SYSTEM_PROMPT, build_prompt


def _flat_to_prompt_dict(flat: dict, heads_config: dict) -> dict:
    """Minimal adapter: flat DB row → dict that build_prompt() expects.

    build_prompt needs: id, gene, score, consequence, substitution, hgvsc, hgvsp,
    disruption {head: [ref, var]}, effect {head: float}, neighbors [...].
    """
    heads = heads_config.get("heads", {})
    disruption = {}
    effect = {}
    for h, info in heads.items():
        if info.get("category") == "disruption":
            ref = flat.get(f"ref_score_{h}")
            var = flat.get(f"var_score_{h}")
            if ref is not None:
                disruption[h] = [ref, var if var is not None else ref]
        else:
            val = flat.get(f"score_{h}")
            if val is not None:
                effect[h] = val

    return {
        "id": flat.get("variant_id", ""),
        "gene": flat.get("gene_name", ""),
        "score": flat.get("score_pathogenic", 0),
        "consequence": flat.get("consequence", ""),
        "substitution": flat.get("substitution", ""),
        "hgvsc": flat.get("hgvsc", ""),
        "hgvsp": flat.get("hgvsp", ""),
        "impact": flat.get("vep_impact", ""),
        "exon": flat.get("exon", ""),
        "label": flat.get("label", ""),
        "loeuf": flat.get("loeuf"),
        "gnomad": flat.get("gnomad"),
        "domains": json.loads(flat["domains"]) if isinstance(flat.get("domains"), str) else flat.get("domains", []),
        "disruption": disruption,
        "effect": effect,
        "gt": {k[3:]: v for k, v in flat.items() if k.startswith("gt_") and isinstance(v, (int, float))},
        "neighbors": json.loads(flat["neighbors"]) if isinstance(flat.get("neighbors"), str) else flat.get("neighbors", []),
    }

# Load .env if present (for ANTHROPIC_API_KEY)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ── Endpoints ─────────────────────────────────────────────────────────


async def global_endpoint(request):
    """GET /api/global — heads config + all distributions (~17KB total)."""
    cur = request.app.state.db.cursor()
    heads_row = cur.execute("SELECT value FROM global_config WHERE key = 'heads'").fetchone()
    dist_row = cur.execute("SELECT value FROM global_config WHERE key = 'distributions'").fetchone()

    result: dict = {}
    if heads_row:
        result["heads"] = json.loads(heads_row[0])
    if dist_row:
        result["distributions"] = json.loads(dist_row[0])
    return JSONResponse(result)


async def umap_endpoint(request):
    """GET /api/umap — UMAP scatter data (or null if not computed)."""
    cur = request.app.state.db.cursor()
    row = cur.execute("SELECT value FROM global_config WHERE key = 'umap'").fetchone()
    if not row:
        return JSONResponse(None)
    return JSONResponse(json.loads(row[0]))



async def variant_endpoint(request):
    """GET /api/variants/{id} — flat row as-is."""
    vid = request.path_params["variant_id"]
    cur = request.app.state.db.cursor()
    row = cur.execute("SELECT * FROM variants WHERE variant_id = ?", [vid]).fetchone()
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)
    columns = [desc[0] for desc in cur.description]
    return JSONResponse({col: val for col, val in zip(columns, row) if val is not None})


async def search_endpoint(request):
    """GET /api/variants/search?q=... — gene prefix search."""
    q = request.query_params.get("q", "").strip().upper()
    if len(q) < 2:
        return JSONResponse([])

    cur = request.app.state.db.cursor()
    results = cur.execute(
        "SELECT variant_id, label_display, score_pathogenic, consequence_display FROM variants "
        "WHERE upper(gene_name) = ? ORDER BY score_pathogenic DESC LIMIT 30",
        [q],
    ).fetchall()

    if len(results) < 30:
        more = cur.execute(
            "SELECT variant_id, label_display, score_pathogenic, consequence_display FROM variants "
            "WHERE upper(gene_name) LIKE ? AND upper(gene_name) != ? "
            "ORDER BY score_pathogenic DESC LIMIT ?",
            [f"{q}%", q, 30 - len(results)],
        ).fetchall()
        results.extend(more)

    return JSONResponse([
        {"v": r[0], "l": r[1], "s": r[2], "c": r[3]} for r in results
    ])


# ── Interpretation ────────────────────────────────────────────────────

_variant_locks: dict[str, asyncio.Lock] = {}


def _get_lock(vid: str) -> asyncio.Lock:
    return _variant_locks.setdefault(vid, asyncio.Lock())


async def interpret_endpoint(request):
    """GET /api/interpret/{id} — on-demand Claude interpretation, cached in DuckDB."""
    vid = request.path_params["variant_id"]
    db_path = request.app.state.db_path

    # 1. Check cache
    cur = request.app.state.db.cursor()
    cached = cur.execute(
        "SELECT variant_id, summary, mechanism, confidence, key_evidence, model, generated_at "
        "FROM interpretations WHERE variant_id = ?",
        [vid],
    ).fetchone()
    if cached:
        cols = [desc[0] for desc in cur.description]
        result = {"status": "ok"}
        for col, val in zip(cols, cached):
            result[col] = json.loads(val) if col == "key_evidence" else val
        return JSONResponse(result)

    # 2. Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return JSONResponse({"status": "unavailable", "error": "ANTHROPIC_API_KEY not set"}, status_code=503)

    # 3. Load variant (flat row → adapt for prompt builder)
    row = cur.execute("SELECT * FROM variants WHERE variant_id = ?", [vid]).fetchone()
    if not row:
        return JSONResponse({"status": "not_found", "error": f"Variant {vid} not found"}, status_code=404)
    columns = [desc[0] for desc in cur.description]
    flat = {col: val for col, val in zip(columns, row) if val is not None}
    variant = _flat_to_prompt_dict(flat, request.app.state.head_config)

    # 4. Generate interpretation (per-variant lock prevents duplicates)
    async with _get_lock(vid):
        # Re-check cache under lock (another request may have completed while we waited)
        cached = cur.execute(
            "SELECT variant_id, summary, mechanism, confidence, key_evidence, model, generated_at "
            "FROM interpretations WHERE variant_id = ?",
            [vid],
        ).fetchone()
        if cached:
            cols = [desc[0] for desc in cur.description]
            result = {"status": "ok"}
            for col, val in zip(cols, cached):
                result[col] = json.loads(val) if col == "key_evidence" else val
            return JSONResponse(result)

        try:
            prompt = build_prompt(variant)
            logger.info(f"Generating interpretation for {vid}...")

            client = anthropic.AsyncAnthropic(api_key=api_key)
            async with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                temperature=0.3,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "summary": {"type": "string"},
                                "mechanism": {"type": "string"},
                                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                "key_evidence": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["summary", "mechanism", "confidence", "key_evidence"],
                            "additionalProperties": False,
                        },
                    }
                },
            ) as stream:
                response = await stream.get_final_message()

            text = next((b.text for b in response.content if b.type == "text"), "{}")
            result = json.loads(text)
            result.update({
                "status": "ok",
                "variant_id": vid,
                "model": "claude-sonnet-4-6",
                "generated_at": time.time(),
            })

            # Cache in DB (same connection, no conflict)
            request.app.state.db.execute(
                "INSERT OR IGNORE INTO interpretations "
                "(variant_id, summary, mechanism, confidence, key_evidence, model, generated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [vid, result["summary"], result["mechanism"], result["confidence"],
                 orjson.dumps(result["key_evidence"]).decode(),
                 result["model"], result["generated_at"]],
            )
            logger.info(f"Cached interpretation for {vid}")

            return JSONResponse(result)

        except Exception as e:
            logger.error(f"Interpretation failed for {vid}: {e}")
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


# ── App factory ───────────────────────────────────────────────────────


def create_app(db_path: Path, static_dir: Path | None = None) -> Starlette:
    """Create the Starlette app with DuckDB-backed API endpoints."""
    routes = [
        Route("/api/global", global_endpoint),
        Route("/api/umap", umap_endpoint),
        Route("/api/variants/search", search_endpoint),
        Route("/api/interpret/{variant_id:path}", interpret_endpoint),
        Route("/api/variants/{variant_id:path}", variant_endpoint),
    ]

    # Serve frontend static files if available
    if static_dir and static_dir.exists():
        from starlette.routing import Mount
        from starlette.staticfiles import StaticFiles
        routes.append(Mount("/", app=StaticFiles(directory=str(static_dir), html=True)))

    middleware = [
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"]),
    ]

    app = Starlette(routes=routes, middleware=middleware)
    app.state.db = duckdb.connect(str(db_path))
    app.state.db_path = db_path

    # Load heads config for the interpret endpoint's prompt builder
    heads_row = app.state.db.execute("SELECT value FROM global_config WHERE key = 'heads'").fetchone()
    app.state.head_config = json.loads(heads_row[0]) if heads_row else {"_meta": {}, "heads": {}}

    return app
