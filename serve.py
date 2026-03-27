"""Lightweight server for the variant viewer with on-demand Claude interpretation.

Serves static files from the build directory and provides a /api/interpret/{variant_id}
endpoint that calls the Claude API to generate variant interpretations on demand.

Usage:
    ANTHROPIC_API_KEY=... uv run --extra serve python serve.py [--build-dir webapp/build_v2] [--port 8080]
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

from display import display_name

# ── Calibration (from pipeline/interpret.py) ──────────────────────────────
CALIBRATION = {
    (0.0, 0.1): 1.8, (0.1, 0.3): 15.6, (0.3, 0.5): 44.4,
    (0.5, 0.7): 69.6, (0.7, 0.9): 88.1, (0.9, 1.01): 96.7,
}


def _calibration_text(score: float) -> str:
    for (lo, hi), rate in CALIBRATION.items():
        if lo <= score < hi:
            return f"Calibration: among labeled variants scoring {lo:.1f}–{hi:.1f}, {rate:.0f}% are pathogenic."
    return ""


SYSTEM_PROMPT = """\
You are a clinical genomics expert interpreting variant pathogenicity predictions \
from Evo2, a 7-billion-parameter DNA foundation model.

Background: Evo2 achieves 0.97 AUROC on a deconfounded ClinVar benchmark, matching \
or exceeding CADD and AlphaMissense across all variant types. ClinVar-trained probes \
generalize to deep mutational scanning experiments (BRCA1, BRCA2, TP53, LDLR), \
confirming the model captures genuine biology.

The system uses a probe trained on reference genome activations, then run on BOTH \
reference and variant activations. For each biological feature (helix, conservation, \
domain membership, etc.), you see:
- **ref**: the probe's prediction on the reference genome (what this position normally looks like)
- **var**: the probe's prediction on the variant genome (what it looks like after mutation)
- **Δ (delta)**: var - ref = the disruption signal

Large negative deltas mean the variant disrupts that feature. Near-zero deltas mean \
the feature is preserved. This is the key signal: pathogenic variants show large \
disruptions, benign variants show near-zero deltas.

A separate **diff probe** predicts what existing clinical tools (CADD, AlphaMissense, \
REVEL, SpliceAI, etc.) would score for this variant. These are NOT external lookups — \
they are Evo2's internal predictions of what those tools would say, based purely on DNA \
sequence context.

An **attribution model** (ridge regression on all probe heads) identifies which heads \
drive the overall pathogenicity prediction. You will see the top contributing heads \
ranked by their contribution to the logit score. Positive contributions push toward \
pathogenic; negative push toward benign.

Provide a clinical interpretation:
- Lead with the disruption story: which features were disrupted (large Δ) and which were preserved
- Use the attribution to explain WHY the model predicts what it does
- Note any disagreements between pathogenicity and predicted clinical scores
- Nearest neighbor consensus is strong independent evidence
- 3-5 sentences for summary, 1 sentence for mechanism"""


# ── Prompt building ───────────────────────────────────────────────────────

def _safe_vid_to_filename(vid: str) -> str:
    """Convert variant_id to safe filename (matching build.py logic)."""
    s = vid.replace(":", "_").replace("/", "_")
    if len(s) <= 200:
        return s
    # FNV-1a 64-bit hash for long indels (same as build.py)
    h = 0xCBF29CE484222325
    for b in vid.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{s[:60]}_{h:016x}"


def build_prompt(v: dict) -> str:
    """Build the interpretation prompt from a per-variant JSON."""
    score = v["score"]
    lines = [f"## {v['gene']} — {v['id']}", v.get('consequence', '?')]
    if v.get('hgvsp') or v.get('hgvsc'):
        parts = []
        if v.get('hgvsp'):
            parts.append(f"Protein: {v['hgvsp']}")
        if v.get('hgvsc'):
            parts.append(f"Coding: {v['hgvsc']}")
        lines.append("  |  ".join(parts))

    # Label context
    label = v.get("label", "?")
    if label != "VUS":
        lines.append(f"ClinVar: {label} ({v.get('significance', '')}), {v.get('stars', '?')} stars")
    else:
        lines.append("ClinVar: VUS (Variant of Uncertain Significance)")
    if v.get("disease"):
        lines.append(f"Disease: {v['disease']}")

    lines.append(f"**Evo2 pathogenicity: {score:.3f}**")
    cal = _calibration_text(score)
    if cal:
        lines.append(cal)
    lines.append("")

    # Top attribution heads (ridge regression coefficients × feature values)
    attr = v.get("attribution") or {}
    attr_heads = attr.get("heads", [])
    disruption = v.get("disruption", {})
    effect = v.get("effect", {})
    gt = v.get("gt", {})

    if attr_heads:
        # Sort by |coefficient| and take top 20
        top = sorted(attr_heads, key=lambda h: abs(h.get("coefficient", 0)), reverse=True)[:20]
        lines.append("### Top Attribution Heads (features most driving the pathogenicity prediction)")
        lines.append("| Rank | Feature | Type | Coefficient | Value | Ref→Var (Δ) | Database |")
        lines.append("|------|---------|------|------------|-------|-------------|----------|")

        for i, h in enumerate(top, 1):
            name = h["name"]
            dname = display_name(name)
            kind = h.get("kind", "?")
            coeff = h.get("coefficient", 0)

            # Get value and ref/var context
            val_str = "—"
            delta_str = "—"
            if kind == "disruption" and name in disruption:
                ref_val, var_val = disruption[name]
                delta = var_val - ref_val
                val_str = f"{var_val:.3f}"
                delta_str = f"{ref_val:.3f}→{var_val:.3f} (Δ={delta:+.3f})"
            elif kind == "effect" and name in effect:
                val_str = f"{effect[name]:.3f}"

            gt_str = f"{gt[name]:.3f}" if name in gt else "—"
            direction = "↑path" if coeff > 0 else "↓benign"
            lines.append(f"| {i} | {dname} | {kind} | {coeff:+.4f} ({direction}) | {val_str} | {delta_str} | {gt_str} |")
        lines.append("")
    else:
        # No attribution model — fall back to top disruptions by |delta|
        top_disruptions = sorted(
            ((name, vals) for name, vals in disruption.items() if isinstance(vals, list) and len(vals) == 2),
            key=lambda x: abs(x[1][1] - x[1][0]), reverse=True,
        )[:15]
        if top_disruptions:
            lines.append("### Top Disrupted Features (largest ref→var changes)")
            lines.append("| Feature | Reference | Variant | Delta | Database |")
            lines.append("|---------|-----------|---------|-------|----------|")
            for name, (ref_val, var_val) in top_disruptions:
                delta = var_val - ref_val
                gt_str = f"{gt[name]:.3f}" if name in gt else "—"
                lines.append(f"| {display_name(name)} | {ref_val:.3f} | {var_val:.3f} | {delta:+.3f} | {gt_str} |")
            lines.append("")

        top_effects = sorted(
            ((name, val) for name, val in effect.items() if name != "pathogenic"),
            key=lambda x: abs(x[1] - 0.5), reverse=True,
        )[:10]
        if top_effects:
            lines.append("### Top Effect Predictions")
            lines.append("| Feature | Predicted | Database |")
            lines.append("|---------|-----------|----------|")
            for name, val in top_effects:
                gt_str = f"{gt[name]:.3f}" if name in gt else "—"
                lines.append(f"| {display_name(name)} | {val:.3f} | {gt_str} |")
            lines.append("")

    # Neighbors
    neighbors = v.get("neighbors", [])
    if neighbors:
        n_p = v.get("nP", 0)
        n_b = v.get("nB", 0)
        n_v = v.get("nV", 0)
        lines.append(f"### Nearest Neighbors: {n_p} pathogenic, {n_b} benign, {n_v} VUS")
        for nb in neighbors[:5]:
            sim = nb.get('similarity', 0)
            lines.append(f"- {nb['gene']} ({nb['label']}, score={nb['score']:.3f}, similarity={sim:.3f})")
        lines.append("")

    # Additional context
    context_parts = []
    if v.get("impact"):
        context_parts.append(f"VEP impact: {v['impact']}")
    if v.get("exon"):
        context_parts.append(f"Exon: {v['exon']}")
    if v.get("domains"):
        context_parts.append(f"Domains: {v['domains']}")
    if v.get("loeuf") is not None:
        context_parts.append(f"LOEUF: {v['loeuf']:.3f}")
    if v.get("gnomad") is not None:
        context_parts.append(f"gnomAD AF: {v['gnomad']:.2e}")
    if context_parts:
        lines.append("### Additional Context")
        lines.extend(f"- {p}" for p in context_parts)

    return "\n".join(lines)


# ── Server ────────────────────────────────────────────────────────────────

BUILD_DIR: Path = Path("webapp/build")
_variant_locks: dict[str, asyncio.Lock] = {}


def _get_lock(vid: str) -> asyncio.Lock:
    if vid not in _variant_locks:
        _variant_locks[vid] = asyncio.Lock()
    return _variant_locks[vid]


async def interpret_endpoint(request):
    """On-demand variant interpretation via Claude API."""
    vid = request.path_params["variant_id"]
    safe = _safe_vid_to_filename(vid)

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
        # Double-check cache after acquiring lock
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

            # Cache to disk
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
        Mount("/", app=StaticFiles(directory=str(build_dir), html=True)),
    ]
    return Starlette(routes=routes)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Variant viewer server with on-demand interpretation")
    parser.add_argument("--build-dir", type=Path, default=Path("webapp/build_v2"))
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
