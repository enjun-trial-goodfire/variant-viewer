"""DuckDB-backed API server for the variant viewer.

Serves variant data, search, global config, UMAP, and on-demand Claude
interpretation from a single DuckDB file. Also serves the Svelte frontend
from frontend/dist/ if present.

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

from db import row_to_interpretation, row_to_variant
from prompts import SYSTEM_PROMPT, build_prompt

# Load .env if present (for ANTHROPIC_API_KEY)
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ── Endpoints ─────────────────────────────────────────────────────────


async def global_endpoint(request):
    """GET /api/global — app config (distributions, heads, display, eval, etc.)."""
    cur = request.app.state.db.cursor()
    rows = cur.execute("SELECT key, value FROM global_config").fetchall()
    result = {}
    for key, value in rows:
        if key == "umap_gene_list":
            continue  # served via /api/umap
        result[key] = json.loads(value)
    return JSONResponse(result)


async def umap_endpoint(request):
    """GET /api/umap — UMAP scatter data."""
    cur = request.app.state.db.cursor()

    count = cur.execute("SELECT COUNT(*) FROM umap").fetchone()[0]
    if count == 0:
        return JSONResponse(None)

    rows = cur.execute(
        "SELECT x, y, score, variant_id, gene_idx, label_idx FROM umap ORDER BY idx"
    ).fetchall()

    gene_list_row = cur.execute(
        "SELECT value FROM global_config WHERE key = 'umap_gene_list'"
    ).fetchone()
    gene_list = json.loads(gene_list_row[0]) if gene_list_row else []

    return JSONResponse({
        "x": [r[0] for r in rows],
        "y": [r[1] for r in rows],
        "score": [r[2] for r in rows],
        "ids": [r[3] for r in rows],
        "genes": [r[4] for r in rows],
        "labels": [r[5] for r in rows],
        "gene_list": gene_list,
    })


async def variant_endpoint(request):
    """GET /api/variants/{id} — single variant lookup."""
    vid = request.path_params["variant_id"]
    cur = request.app.state.db.cursor()
    row = cur.execute("SELECT * FROM variants WHERE variant_id = ?", [vid]).fetchone()
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)
    columns = [desc[0] for desc in cur.description]
    return JSONResponse(row_to_variant(row, columns))


async def search_endpoint(request):
    """GET /api/variants/search?q=... — gene prefix search."""
    q = request.query_params.get("q", "").strip().upper()
    if len(q) < 2:
        return JSONResponse([])

    cur = request.app.state.db.cursor()

    # Exact gene match first
    results = cur.execute(
        "SELECT variant_id, label, score, consequence FROM variants "
        "WHERE upper(gene) = ? ORDER BY score DESC LIMIT 30",
        [q],
    ).fetchall()

    # Prefix match if we need more
    if len(results) < 30:
        prefix_results = cur.execute(
            "SELECT variant_id, label, score, consequence FROM variants "
            "WHERE upper(gene) LIKE ? AND upper(gene) != ? "
            "ORDER BY score DESC LIMIT ?",
            [f"{q}%", q, 30 - len(results)],
        ).fetchall()
        results.extend(prefix_results)

    return JSONResponse([
        {"v": r[0], "l": r[1], "s": r[2], "c": r[3]} for r in results
    ])


# ── Interpretation ────────────────────────────────────────────────────

_interpret_lock = asyncio.Lock()


async def interpret_endpoint(request):
    """GET /api/interpret/{id} — on-demand Claude interpretation, cached in DuckDB."""
    vid = request.path_params["variant_id"]
    db_path = request.app.state.db_path

    # 1. Check cache (read-only connection)
    cur = request.app.state.db.cursor()
    cached = cur.execute(
        "SELECT variant_id, summary, mechanism, confidence, key_evidence, model, generated_at "
        "FROM interpretations WHERE variant_id = ?",
        [vid],
    ).fetchone()
    if cached:
        columns = [desc[0] for desc in cur.description]
        return JSONResponse(row_to_interpretation(cached, columns))

    # 2. Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return JSONResponse({"status": "unavailable", "error": "ANTHROPIC_API_KEY not set"}, status_code=503)

    # 3. Load variant
    row = cur.execute("SELECT * FROM variants WHERE variant_id = ?", [vid]).fetchone()
    if not row:
        return JSONResponse({"status": "not_found", "error": f"Variant {vid} not found"}, status_code=404)
    columns = [desc[0] for desc in cur.description]
    variant = row_to_variant(row, columns)

    # 4. Generate interpretation (with lock to prevent duplicates)
    async with _interpret_lock:
        # Re-check cache under lock
        with duckdb.connect(str(db_path), read_only=True) as check_conn:
            cached = check_conn.execute(
                "SELECT variant_id, summary, mechanism, confidence, key_evidence, model, generated_at "
                "FROM interpretations WHERE variant_id = ?",
                [vid],
            ).fetchone()
            if cached:
                columns = [desc[0] for desc in check_conn.cursor().description]
                return JSONResponse(row_to_interpretation(cached, ["variant_id", "summary", "mechanism", "confidence", "key_evidence", "model", "generated_at"]))

        try:
            import anthropic

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

            # Write to DB
            with duckdb.connect(str(db_path)) as write_conn:
                write_conn.execute(
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
    app.state.db = duckdb.connect(str(db_path), read_only=True)
    app.state.db_path = db_path
    return app
