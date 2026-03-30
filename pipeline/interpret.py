"""Generate clinical interpretations for variants using Claude API.

Builds structured prompts from raw probe scores + ground truth annotations,
calls Claude, saves JSON. Uses batch mode for multiple variants.

Usage:
    uv run python pipeline/interpret.py --variant chr12:5575980:G:A --dry-run
    uv run python pipeline/interpret.py --variant chr12:5575980:G:A
    uv run python pipeline/interpret.py --top-vus 50 --concurrency 5

Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import polars as pl
import torch
import typer
from loguru import logger

from attribution import attribute
from constants import CONSEQUENCE_CLASSES, AA_SWAP_CLASSES
from paths import ARTIFACTS, MAYO_DATA
from loaders import load_metadata
from prompts import SYSTEM_PROMPT, build_prompt

LABELED_ACT = ARTIFACTS / "clinvar_evo2_deconfounded_full"
VUS_ACT = ARTIFACTS / "clinvar_evo2_vus"
PROBE = "probe_v9"
OUTPUT_DIR = VUS_ACT / "interpretations"

CSQ_LABELS = {i: name for i, name in enumerate(CONSEQUENCE_CLASSES)}
AA_LABELS = {i: name for i, name in enumerate(AA_SWAP_CLASSES)}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load all data needed for interpretation: scores, metadata, attribution, embeddings."""
    labeled_scores = pl.read_ipc(str(LABELED_ACT / PROBE / "scores.feather"))
    vus_scores = pl.read_ipc(str(VUS_ACT / PROBE / "scores.feather"))

    labeled_meta = load_metadata("deconfounded-full")
    vus_meta = load_metadata("vus")

    gt_annotations = pl.read_ipc(str(MAYO_DATA / "clinvar" / "deconfounded-full" / "annotations.feather"))

    meta_cols = ("variant_id", "label", "consequence", "gene_name",
                 "clinical_significance", "stars", "disease_name")

    labeled = labeled_scores.join(
        labeled_meta.select(*meta_cols), on="variant_id", how="left"
    ).with_columns(pl.col("stars").cast(pl.Int32))

    gt_cols = [c for c in gt_annotations.columns if c != "variant_id"]
    labeled = labeled.join(
        gt_annotations.rename({c: f"gt_{c}" for c in gt_cols}),
        left_on="variant_id", right_on="variant_id", how="left",
    )

    vus = vus_scores.join(
        vus_meta.select(*(c for c in meta_cols if c not in ("label", "stars"))),
        on="variant_id", how="left",
    ).with_columns(pl.lit("VUS").alias("label"), pl.lit(0).cast(pl.Int32).alias("stars"))

    all_v = pl.concat([labeled, vus], how="diagonal")

    if "pred_consequence" in all_v.columns:
        all_v = all_v.with_columns(
            pl.col("pred_consequence").replace_strict(CSQ_LABELS, default="unknown").alias("csq"),
            pl.col("pred_aa_swap").replace_strict(AA_LABELS, default=None).alias("aa"),
        )

    idx = {vid: i for i, vid in enumerate(all_v["variant_id"].to_list())}

    # Probe config for head classification
    cfg = json.loads((LABELED_ACT / PROBE / "config.json").read_text())
    disruption_set = set(cfg.get("disruption_heads", cfg.get("ref_heads", ())))
    effect_set = set(cfg.get("effect_heads", cfg.get("diff_heads", ())))

    # Attribution (z-scored disruption deltas)
    disruption_heads = tuple(cfg.get("disruption_heads", cfg.get("ref_heads", ())))
    attr_by_vid = attribute(all_v, disruption_heads, k=10)

    # Embeddings
    from goodfire_core.storage import ActivationDataset, FilesystemStorage

    def load_emb(path):
        storage = FilesystemStorage(path / PROBE)
        ds = ActivationDataset(storage, "embeddings", batch_size=4096, include_provenance=True)
        it = ds.training_iterator(device="cpu", n_epochs=1, shuffle=False, drop_last=False)
        embs, ids = [], []
        for batch in it.iter_epoch():
            embs.append(batch.acts.flatten(1))
            ids.extend(batch.sequence_ids)
        return torch.cat(embs), ids

    emb_l, ids_l = load_emb(LABELED_ACT)
    emb_v, ids_v = load_emb(VUS_ACT)
    emb_all = torch.nn.functional.normalize(torch.cat([emb_l, emb_v]).float(), dim=1)
    emb_ids = ids_l + ids_v
    emb_idx = {vid: i for i, vid in enumerate(emb_ids)}

    return {
        "all": all_v, "idx": idx,
        "disruption_set": disruption_set, "effect_set": effect_set,
        "attr_by_vid": attr_by_vid,
        "emb_all": emb_all, "emb_ids": emb_ids, "emb_idx": emb_idx,
    }


def get_neighbors(vid: str, data: dict, k: int = 10) -> list[dict]:
    """Find k nearest neighbors by embedding cosine similarity."""
    if vid not in data["emb_idx"]:
        return []
    q = data["emb_all"][data["emb_idx"][vid]].unsqueeze(0)
    sims = (q @ data["emb_all"].T).squeeze(0)
    sims[data["emb_idx"][vid]] = -1
    topk = sims.topk(k)
    result = []
    for i, sim in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        nid = data["emb_ids"][i]
        if nid not in data["idx"]:
            continue
        r = data["all"].row(data["idx"][nid], named=True)
        result.append({
            "id": nid, "gene": r.get("gene_name", "?"),
            "label": r.get("label", "?"),
            "score": round(r.get("score_pathogenic", 0), 3),
            "similarity": round(sim, 3),
        })
    return result


def row_to_variant(row: dict, data: dict, neighbors: list[dict]) -> dict:
    """Convert a DataFrame row dict to the variant JSON schema expected by prompts.build_prompt."""
    vid = row["variant_id"]
    disruption_set = data["disruption_set"]
    effect_set = data["effect_set"]

    # Disruption: compute deltas from ref/var score columns
    disruption = {}
    for key in row:
        if not key.startswith("ref_score_"):
            continue
        h = key[10:]  # strip "ref_score_"
        if h not in disruption_set:
            continue
        ref_val = row[key]
        var_val = row.get(f"var_score_{h}")
        if ref_val is None or var_val is None:
            continue
        delta = round(var_val - ref_val, 4)
        if abs(delta) > 0.01:
            disruption[h] = delta

    # Effect: raw diff scores
    effect = {}
    for key in row:
        if not key.startswith("score_") or key == "score_pathogenic":
            continue
        h = key[6:]  # strip "score_"
        if h not in effect_set:
            continue
        val = row[key]
        if val is not None:
            effect[h] = val

    # Ground truth annotations
    gt = {}
    for key in row:
        if not key.startswith("gt_"):
            continue
        val = row[key]
        if val is not None and isinstance(val, (int, float)) and val > 0:
            gt[key[3:]] = val

    # Attribution (already a list of {name, z} dicts from attribute())
    attribution = data["attr_by_vid"].get(vid, [])

    n_p = sum(1 for nb in neighbors if "pathogenic" in nb.get("label", ""))
    n_b = sum(1 for nb in neighbors if "benign" in nb.get("label", ""))

    return {
        "id": vid,
        "gene": row.get("gene_name", "?"),
        "consequence": row.get("csq") or row.get("consequence", "?"),
        "substitution": row.get("aa"),
        "label": row.get("label", "?"),
        "significance": row.get("clinical_significance", ""),
        "stars": row.get("stars", 0),
        "disease": row.get("disease_name") or "",
        "score": row.get("score_pathogenic", 0),
        "disruption": disruption,
        "effect": effect,
        "gt": gt,
        "attribution": attribution,
        "neighbors": neighbors,
        "nP": n_p, "nB": n_b, "nV": len(neighbors) - n_p - n_b,
    }


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------
async def interpret_one(
    vid: str, data: dict, client, dry_run: bool = False,
) -> dict:
    """Interpret a single variant: build prompt, call Claude, return result."""
    row = data["all"].row(data["idx"][vid], named=True)
    neighbors = get_neighbors(vid, data)
    variant = row_to_variant(row, data, neighbors)
    prompt = build_prompt(variant)

    if dry_run:
        print(f"\n{'='*80}\n{prompt}\n{'='*80}")
        return {"variant_id": vid, "status": "dry_run"}

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
                        "ground_truth_agreement": {"type": "string"},
                    },
                    "required": ["summary", "mechanism", "confidence", "key_evidence", "ground_truth_agreement"],
                    "additionalProperties": False,
                },
            }
        },
    ) as stream:
        response = await stream.get_final_message()

    text = next((b.text for b in response.content if b.type == "text"), "{}")
    result = json.loads(text)
    result.update({
        "variant_id": vid,
        "gene_name": variant["gene"],
        "score": round(variant["score"], 4),
        "label": variant["label"],
        "model": "claude-sonnet-4-6",
        "generated_at": time.time(),
        "status": "ok",
    })
    return result


async def batch(vids: list[str], data: dict, concurrency: int, dry_run: bool) -> list[dict]:
    import anthropic
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def go(vid):
        async with sem:
            logger.info(f"Interpreting {vid}...")
            r = await interpret_one(vid, data, client, dry_run)
            results.append(r)
            if r.get("status") == "ok":
                logger.info(f"  → {r.get('confidence', '?')} confidence")

    await asyncio.gather(*(go(v) for v in vids))
    return results


def main(
    variant: Optional[str] = typer.Option(None, help="Single variant ID"),
    top_vus: Optional[int] = typer.Option(None, help="Top N VUS by score"),
    concurrency: int = typer.Option(5, help="Max concurrent API calls"),
    dry_run: bool = typer.Option(False, help="Print prompts without calling API"),
    output: Path = typer.Option(OUTPUT_DIR, help="Output directory for interpretations"),
) -> None:
    if not variant and not top_vus:
        raise typer.BadParameter("Specify --variant or --top-vus")

    logger.info("Loading data...")
    data = load_data()
    logger.info(f"Loaded {data['all'].height:,} variants, {len(data['attr_by_vid']):,} attributions")

    if variant:
        vids = [variant]
    else:
        vus = data["all"].filter(pl.col("label") == "VUS")
        vids = vus.sort("score_pathogenic", descending=True).head(top_vus)["variant_id"].to_list()

    # Cost guard: batch interpretation is expensive (~$0.01/variant, $500+ for 50K VUS)
    if len(vids) > 20 and not dry_run:
        cost_est = len(vids) * 0.01
        logger.warning(f"About to interpret {len(vids):,} variants (~${cost_est:,.0f} API cost)")
        confirm = input(f"Continue with {len(vids)} variants? [y/N] ").strip().lower()
        if confirm != "y":
            logger.info("Aborted.")
            raise typer.Exit(0)

    results = asyncio.run(batch(vids, data, concurrency, dry_run))

    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r.get("status") != "ok":
                continue
            safe = r["variant_id"].replace(":", "_")
            (output / f"{safe}.json").write_text(json.dumps(r, indent=2))
        (output / "summary.json").write_text(json.dumps(results, indent=2))
        logger.info(f"Saved {len(results)} interpretations to {output}")


if __name__ == "__main__":
    typer.run(main)
