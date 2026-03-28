"""Lightweight server for the variant viewer with on-demand Claude interpretation.

Serves static files from the build directory and provides a /api/interpret/{variant_id}
endpoint that calls the Claude API to generate variant interpretations on demand.

Usage:
    ANTHROPIC_API_KEY=... uv run --extra serve python serve.py [--build-dir webapp/build] [--port 8080]
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import orjson
from loguru import logger
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from constants import CALIBRATION
from paths import sanitize_vid
from prompts import SYSTEM_PROMPT, build_prompt

# ── Server state ──────────────────────────────────────────────────────────

BUILD_DIR: Path = Path("webapp/build")
_variant_locks: dict[str, asyncio.Lock] = {}


def _get_lock(vid: str) -> asyncio.Lock:
    if vid not in _variant_locks:
        _variant_locks[vid] = asyncio.Lock()
    return _variant_locks[vid]


async def interpret_endpoint(request):
    """On-demand variant interpretation via Claude API."""
    vid = request.path_params["variant_id"]
    safe = sanitize_vid(vid)

    # 1. Check cache
    cache_dir = BUILD_DIR / "interpretations"
    cache_file = cache_dir / f"{safe}.json"
    if cache_file.exists():
        return Response(cache_file.read_bytes(), media_type="application/json")

    # 2. Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return JSONResponse({"status": "unavailable", "error": "ANTHROPIC_API_KEY not set"}, status_code=503)

    # 3. Load variant JSON
    variant_file = BUILD_DIR / "variants" / f"{safe}.json"
    if not variant_file.exists():
        return JSONResponse({"status": "not_found", "error": f"Variant {vid} not found"}, status_code=404)

    # 4. Generate (with per-variant lock to prevent duplicates)
    async with _get_lock(vid):
        if cache_file.exists():
            return Response(cache_file.read_bytes(), media_type="application/json")

        try:
            import anthropic

            v = orjson.loads(variant_file.read_bytes())
            prompt = build_prompt(v)
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

            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(orjson.dumps(result))
            logger.info(f"Cached interpretation for {vid}")

            return JSONResponse(result)

        except Exception as e:
            logger.error(f"Interpretation failed for {vid}: {e}")
            return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


def create_app(build_dir: Path) -> Starlette:
    global BUILD_DIR
    BUILD_DIR = build_dir

    routes = [
        Route("/api/interpret/{variant_id:path}", interpret_endpoint),
        Mount("/", app=StaticFiles(directory=str(build_dir), html=True, follow_symlink=True)),
    ]
    return Starlette(routes=routes)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Variant viewer server with on-demand interpretation")
    parser.add_argument("--build-dir", type=Path, default=Path("webapp/build"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if not args.build_dir.exists():
        logger.error(f"Build directory not found: {args.build_dir}")
        return

    app = create_app(args.build_dir)
    logger.info(f"Serving {args.build_dir} on http://{args.host}:{args.port}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — interpretation endpoint will return 503")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
