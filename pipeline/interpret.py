"""Generate clinical interpretations for variants using Claude API.

Builds structured prompts from raw probe scores + ground truth annotations,
calls Claude, saves JSON. Uses batch mode for multiple variants.

Usage:
    uv run python pipeline/interpret.py --variant chr12:5575980:G:A --dry-run
    uv run python pipeline/interpret.py --variant chr12:5575980:G:A
    uv run python pipeline/interpret.py --top-vus 50 --concurrency 5

Requires ANTHROPIC_API_KEY environment variable.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import anthropic
import numpy as np
import polars as pl
import torch
from loguru import logger

from constants import CONSEQUENCE_CLASSES, AA_SWAP_CLASSES, CALIBRATION, calibration_text
from paths import ARTIFACTS, MAYO_DATA, load_metadata

LABELED_ACT = ARTIFACTS / "clinvar_evo2_deconfounded_full"
VUS_ACT = ARTIFACTS / "clinvar_evo2_vus"
PROBE = "probe_v9"
OUTPUT_DIR = VUS_ACT / "interpretations"

CSQ_LABELS = {i: name for i, name in enumerate(CONSEQUENCE_CLASSES)}
AA_LABELS = {i: name for i, name in enumerate(AA_SWAP_CLASSES)}

PFAM_NAMES_V3: dict[str, str] = {
    "PF00069": "Protein kinase", "PF00520": "Ion transport", "PF07714": "Tyr kinase",
    "PF01391": "Collagen", "PF00067": "Cytochrome P450", "PF00041": "Fibronectin III",
    "PF00630": "Filamin", "PF00028": "Cadherin", "PF00063": "Myosin head",
    "PF00038": "Intermediate filament", "PF00089": "Trypsin", "PF00001": "GPCR (7tm_1)",
    "PF00884": "Sulfatase", "PF01576": "Myosin tail", "PF07679": "Ig I-set",
    "PF00092": "von Willebrand A", "PF00022": "Actin", "PF00209": "Na/K channel",
    "PF00435": "Spectrin repeat", "PF00096": "Zinc finger C2H2",
    "PF00400": "WD40 repeat", "PF00079": "Serpin", "PF00005": "ABC transporter",
    "PF13853": "Leucine-rich repeat", "PF00071": "Ras GTPase",
    "PF00083": "Sugar transporter", "PF00664": "ABC transporter TM",
    "PF00029": "Lectin C-type", "PF07645": "EGF calcium-binding",
    "PF07690": "Major facilitator superfamily", "PF00008": "EGF-like",
    "PF00270": "DEAD box helicase", "PF00171": "Aldehyde dehydrogenase",
    "PF07686": "Ig V-set", "PF00654": "Voltage-gated channel",
    "PF00057": "Low-density lipoprotein receptor A", "PF00046": "Homeobox",
    "PF00271": "Helicase C-terminal", "PF00755": "Chondroitin sulfate synthase",
    "PF01496": "V-type ATPase", "PF00916": "Sulfate transporter",
    "PF04388": "Potassium channel tetramerisation", "PF00206": "Lyase aromatic",
    "PF08423": "PIGA (GPI anchor)", "PF02932": "Neurotransmitter-gated channel TM",
    "PF16499": "Cadherin tandem repeat", "PF00053": "Laminin EGF-like",
    "PF00995": "Sec1/Munc18", "PF00153": "Mitochondrial carrier",
    "PF00176": "SNF2 ATPase", "PF00004": "AAA ATPase", "PF12796": "Ankyrin repeat",
    "PF01055": "Glycosyl hydrolase 31", "PF13927": "Ig-like",
    "PF00149": "Calcineurin-like phosphoesterase", "PF00225": "Kinesin motor",
    "PF00245": "Metalloproteinase", "PF00110": "Wnt",
    "PF02931": "Neurotransmitter-gated channel ligand-binding",
    "PF11864": "Domain of unknown function (DUF3408)",
    "PF01094": "Receptor family ligand binding", "PF00688": "TBC Rab-GAP",
    "PF01410": "Fibrillar collagen C-terminal", "PF00118": "TCP-1/cpn60 chaperonin",
    "PF00122": "E1-E2 ATPase", "PF03542": "Cytochrome P450 (N-terminal)",
    "PF00084": "Sushi/SCR/CCP", "PF00091": "Tubulin",
    "PF00441": "Acyl-CoA dehydrogenase (C-terminal)", "PF00651": "BTB/POZ",
    "PF02463": "RecF/RecN/SMC", "PF01039": "Carboxylesterase",
    "PF00870": "P450 reductase", "PF24681": "Zinc finger RanBP2",
    "PF00058": "Gamma-carboxyglutamic acid (Gla)", "PF00266": "Aminotransferase",
    "PF00017": "SH2", "PF01347": "Nucleoside diphosphate kinase",
    "PF00136": "DNA polymerase family B", "PF01602": "Adaptin N-terminal",
    "PF00343": "Phospholipase C (catalytic)", "PF00501": "AMP-binding enzyme",
    "PF00412": "LIM domain",
}

HEAD_DISPLAY = {
    "phylop_c": "PhyloP conservation", "phastcons_c": "PhastCons",
    "blosum62_c": "BLOSUM62", "grantham_c": "Grantham distance",
    "af_helix": "Alpha helix", "af_strand": "Beta strand",
    "af_buried": "Buried residue", "af_confidence_c": "AlphaFold pLDDT",
    "in_transmembrane": "Transmembrane", "in_binding_site": "Binding site",
    "in_active_site": "Active site", "is_disordered": "Disordered",
    "has_ptm": "PTM site", "ppi_interface": "PPI interface",
    "in_domain": "Protein domain", "is_exonic": "Exonic",
    "in_promoter": "Promoter", "in_enhancer": "Enhancer",
    "cpg_island": "CpG island", "remm_c": "ReMM regulatory",
    "splice_disrupting": "Splice disrupting", "charge_altering": "Charge altering",
    "loeuf_c": "Gene constraint (LOEUF)", "in_repeat": "Repeat element",
    "cadd_c": "CADD", "alphamissense_c": "AlphaMissense",
    "sift_c": "SIFT", "polyphen_c": "PolyPhen-2",
    "eve_c": "EVE", "vest4_c": "VEST4", "gerp_c": "GERP", "mpc_c": "MPC",
    "gnomad_af_c": "gnomAD allele frequency",
}

HEAD_GROUPS = {
    "Clinical predictors": ("cadd_c", "alphamissense_c", "sift_c", "polyphen_c",
                             "eve_c", "vest4_c", "mpc_c"),
    "Conservation": ("phylop_c", "phastcons_c", "gerp_c", "blosum62_c", "grantham_c"),
    "Protein structure": ("af_helix", "af_strand", "af_buried", "af_confidence_c",
                           "in_transmembrane", "in_binding_site", "in_active_site",
                           "is_disordered", "has_ptm", "ppi_interface"),
    "Domain": ("in_domain", "is_exonic"),
    "Regulatory": ("in_promoter", "in_enhancer", "cpg_island", "remm_c"),
    "Genomic context": ("splice_disrupting", "charge_altering", "loeuf_c",
                         "gnomad_af_c", "in_repeat"),
}

# Calibration: % of variants that are actually pathogenic in each score bin.
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
sequence context. This is powerful because:

- When pathogenicity is HIGH but predicted CADD/AlphaMissense are LOW, note the disagreement \
objectively — the DNA model detects something these residue-level tools may not capture.
- When predictions and clinical scores agree, that's consensus worth noting.
- When predicted clinical scores are high but pathogenicity is low, mention the discrepancy \
without forcing an explanation.

Provide a clinical interpretation:
- Lead with the disruption story: which features were disrupted (large Δ) and which were preserved
- Use the position context (ref scores) to explain WHY disruption matters (disrupting a helix \
in a transmembrane domain is worse than disrupting a helix in a disordered region)
- Note any disagreements between pathogenicity and predicted clinical scores — \
these are scientifically interesting regardless of which is correct
- Nearest neighbor consensus is strong independent evidence
- 3-5 sentences for summary, 1 sentence for mechanism"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    # Load unified scores from probe_v6 (score_* for diff, ref_score_*/var_score_* for ref)
    labeled_scores = pl.read_ipc(str(LABELED_ACT / PROBE / "scores.feather"))
    vus_scores = pl.read_ipc(str(VUS_ACT / PROBE / "scores.feather"))

    labeled_meta = load_metadata("deconfounded-full")
    vus_meta = load_metadata("vus")

    # Ground truth annotations
    gt_annotations = pl.read_ipc(str(MAYO_DATA / "clinvar" / "deconfounded-full" / "annotations.feather"))

    meta_cols = ("variant_id", "label", "consequence", "gene_name",
                 "clinical_significance", "stars", "disease_name")

    labeled = labeled_scores.join(
        labeled_meta.select(*meta_cols), on="variant_id", how="left"
    ).with_columns(pl.col("stars").cast(pl.Int32))

    # Join ground truth to labeled
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

    # Map prediction indices to names
    if "pred_consequence" in all_v.columns:
        all_v = all_v.with_columns(
            pl.col("pred_consequence").replace_strict(CSQ_LABELS, default="unknown").alias("csq"),
            pl.col("pred_aa_swap").replace_strict(AA_LABELS, default=None).alias("aa"),
        )

    idx = {vid: i for i, vid in enumerate(all_v["variant_id"].to_list())}

    baselines = {}
    for col in [c for c in labeled_scores.columns if "score_" in c]:
        baselines[col] = np.sort(labeled[col].drop_nulls().to_numpy())

    # Domain pathogenic rates (for prompt context)
    domain_rates = {}
    pfam_cols = [c for c in gt_annotations.columns if c.startswith("pfam_") and "_x_path" not in c]
    for col in pfam_cols:
        in_domain = gt_annotations.filter(pl.col(col) == 1)
        if in_domain.height < 20:
            continue
        joined = in_domain.join(labeled_meta.select("variant_id", "label"), on="variant_id", how="left")
        rate = (joined["label"] == "pathogenic").mean()
        domain_rates[col] = round(rate * 100, 1)

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

    return {"all": all_v, "idx": idx, "baselines": baselines,
            "domain_rates": domain_rates,
            "emb_all": emb_all, "emb_ids": emb_ids, "emb_idx": emb_idx}


def pctl(value: float, col: str, baselines: dict) -> float:
    ref = baselines.get(col)
    if ref is None or len(ref) == 0:
        return 50.0
    return round(100.0 * np.searchsorted(ref, value) / len(ref), 1)


def get_neighbors(vid: str, data: dict, k: int = 10) -> list[dict]:
    if vid not in data["emb_idx"]:
        return []
    q = data["emb_all"][data["emb_idx"][vid]].unsqueeze(0)
    sims = (q @ data["emb_all"].T).squeeze(0)
    sims[data["emb_idx"][vid]] = -1
    topk = sims.topk(k)
    result = []
    for i, _s in zip(topk.indices.tolist(), topk.values.tolist(), strict=True):
        nid = data["emb_ids"][i]
        if nid not in data["idx"]:
            continue
        r = data["all"].row(data["idx"][nid], named=True)
        result.append({"variant_id": nid, "gene": r.get("gene_name", "?"),
                        "label": r.get("label", "?"), "score": round(r.get("score_pathogenic", 0), 3)})
    return result


def _gt_value(row: dict, head: str) -> float | None:
    """Get ground truth annotation for a head, or None if missing."""
    gt_col = f"gt_{head}"
    val = row.get(gt_col)
    if val is None:
        return None
    fval = float(val)
    if np.isnan(fval) or fval < 0:
        return None
    return fval


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def _format_heads(row: dict, heads: tuple, prefix: str, data: dict) -> list[str]:
    """Format a group of heads as prompt lines."""
    items = []
    for h in heads:
        col = f"{prefix}score_{h}"
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        pp = pctl(v, col, data["baselines"])
        flag = " **" if pp > 90 or pp < 10 else ""
        gt = _gt_value(row, h)
        gt_text = f" [gt={gt:.2f}]" if gt is not None else ""
        items.append(f"- {HEAD_DISPLAY.get(h, h)}: {v:.3f} (p{pp:.0f}){gt_text}{flag}")
    return items


def build_prompt(vid: str, data: dict) -> str:
    row = data["all"].row(data["idx"][vid], named=True)
    score = row["score_pathogenic"]
    p = pctl(score, "score_pathogenic", data["baselines"])

    lines = [
        f"## {row.get('gene_name', '?')} — {vid}",
        f"{row.get('csq', '?')}" + (f" ({row.get('aa')})" if row.get("aa") else ""),
        f"ClinVar: {row.get('label', '?')}" + (f" ({row.get('clinical_significance', '')})" if row.get("label") != "VUS" else ""),
        f"Disease: {row.get('disease_name') or 'None'}",
        f"**Evo2 pathogenicity: {score:.3f} (p{p:.0f})**",
        calibration_text(score),
        "",
    ]

    # Reference probe: position context
    ref_groups = {
        "Conservation": ("phylop_c", "phastcons_c", "gerp_c"),
        "Protein structure": ("af_helix", "af_strand", "af_buried", "af_confidence_c",
                               "in_transmembrane", "in_binding_site", "in_active_site",
                               "is_disordered", "has_ptm", "ppi_interface"),
        "Domain": ("in_domain", "is_exonic"),
        "Regulatory": ("in_promoter", "in_enhancer", "cpg_island", "remm_c"),
        "Genomic context": ("gc_content_c", "mutation_rate_c", "recomb_rate_c", "loeuf_c", "in_repeat"),
    }

    lines.append("### Position Context + Disruption")
    lines.append("(ref = reference genome, var = variant genome, Δ = disruption)")
    for group, heads in ref_groups.items():
        items = []
        for h in heads:
            ref_val = row.get(f"ref_score_{h}")
            var_val = row.get(f"var_score_{h}")
            if ref_val is None or (isinstance(ref_val, float) and np.isnan(ref_val)):
                continue
            name = HEAD_DISPLAY.get(h, h)
            if var_val is not None and not (isinstance(var_val, float) and np.isnan(var_val)):
                delta = var_val - ref_val
                flag = " **" if abs(delta) > 0.1 else ""
                items.append(f"- {name}: ref={ref_val:.3f} → var={var_val:.3f} (Δ={delta:+.3f}){flag}")
            else:
                items.append(f"- {name}: ref={ref_val:.3f}")
        if items:
            lines.append(f"**{group}:**")
            lines.extend(items)

    # Active Pfam domains with disruption
    pfam_items = []
    for col_name in sorted(row.keys()):
        if not col_name.startswith("ref_score_pfam_"):
            continue
        ref_val = row.get(col_name)
        if ref_val is None or (isinstance(ref_val, float) and np.isnan(ref_val)) or ref_val < 0.1:
            continue
        pfam_id = col_name.replace("ref_score_pfam_", "")
        name = PFAM_NAMES_V3.get(pfam_id, pfam_id) if PFAM_NAMES_V3 else pfam_id
        var_val = row.get(f"var_score_pfam_{pfam_id}")
        rate = data["domain_rates"].get(f"pfam_{pfam_id}")
        rate_text = f" ({rate:.0f}% pathogenic)" if rate else ""
        if var_val is not None and not (isinstance(var_val, float) and np.isnan(var_val)):
            delta = var_val - ref_val
            pfam_items.append(f"- {name}: ref={ref_val:.3f} → var={var_val:.3f} (Δ={delta:+.3f}){rate_text}")
        else:
            pfam_items.append(f"- {name}: {ref_val:.3f}{rate_text}")
    if pfam_items:
        lines.append("**Protein domains:**")
        lines.extend(pfam_items)

    # Diff probe: variant effects
    diff_groups = {
        "Clinical predictors": ("cadd_c", "alphamissense_c", "sift_c", "polyphen_c",
                                 "eve_c", "vest4_c", "mpc_c"),
        "Substitution properties": ("blosum62_c", "grantham_c"),
        "Variant effects": ("splice_disrupting", "charge_altering", "gnomad_af_c"),
    }

    lines.append("\n### Variant Effects (diff probe)")
    for group, heads in diff_groups.items():
        items = _format_heads(row, heads, "", data)
        if items:
            lines.append(f"**{group}:**")
            lines.extend(items)

    # Neighbors
    nb = get_neighbors(vid, data)
    n_p = sum(1 for n in nb if "pathogenic" in n["label"])
    n_b = sum(1 for n in nb if "benign" in n["label"])
    n_v = sum(1 for n in nb if n["label"] == "VUS")
    lines.append(f"\n**Nearest neighbors:** {n_p} pathogenic, {n_b} benign, {n_v} VUS")
    lines.extend(f"- {n['gene']} ({n['label']}, score={n['score']})" for n in nb[:5])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------
async def interpret_one(
    vid: str, data: dict, client: anthropic.AsyncAnthropic, dry_run: bool = False,
) -> dict:
    prompt = build_prompt(vid, data)
    if dry_run:
        print(f"\n{'='*80}\n{prompt}\n{'='*80}")
        return {"variant_id": vid, "status": "dry_run"}

    row = data["all"].row(data["idx"][vid], named=True)

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
        "gene_name": row.get("gene_name", "?"),
        "score": round(row["score_pathogenic"], 4),
        "label": row.get("label", "?"),
        "model": "claude-sonnet-4-6",
        "generated_at": time.time(),
        "status": "ok",
    })
    return result


async def batch(vids: list[str], data: dict, concurrency: int, dry_run: bool) -> list[dict]:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", help="Single variant ID")
    parser.add_argument("--top-vus", type=int, help="Top N VUS by score")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    logger.info("Loading data...")
    data = load_data()
    logger.info(f"Loaded {data['all'].height:,} variants, {len(data['domain_rates'])} domain stats")

    if args.variant:
        vids = [args.variant]
    elif args.top_vus:
        vus = data["all"].filter(pl.col("label") == "VUS")
        vids = vus.sort("score_pathogenic", descending=True).head(args.top_vus)["variant_id"].to_list()
    else:
        parser.error("Specify --variant or --top-vus")
        return

    results = asyncio.run(batch(vids, data, args.concurrency, args.dry_run))

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r.get("status") != "ok":
                continue
            safe = r["variant_id"].replace(":", "_")
            (args.output / f"{safe}.json").write_text(json.dumps(r, indent=2))
        (args.output / "summary.json").write_text(json.dumps(results, indent=2))
        logger.info(f"Saved {len(results)} interpretations to {args.output}")


if __name__ == "__main__":
    main()
