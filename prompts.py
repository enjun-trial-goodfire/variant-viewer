"""Claude interpretation prompts for the variant viewer.

Single source of truth for the system prompt and prompt builder.
Used by serve.py (on-demand) and pipeline/interpret.py (batch).
"""

from pathlib import Path

from constants import calibration_text
from display import _EFFECT_PREDICTOR_HEADS, curated_group, display_name

SYSTEM_PROMPT = """\
You are a clinical genomics expert interpreting variant pathogenicity predictions \
from Evo2, a 7-billion-parameter DNA foundation model.

The system uses a probe trained on reference genome activations, then run on BOTH \
reference and variant activations. For each biological feature (helix, conservation, \
domain membership, etc.), you see:
- **ref**: the probe's prediction on the reference genome (what this position normally looks like)
- **var**: the probe's prediction on the variant genome (what it looks like after mutation)
- **Δ (delta)**: var - ref = the disruption signal

Large negative deltas mean the variant disrupts that feature. Near-zero deltas mean \
the feature is preserved. This is the key signal: pathogenic variants show large \
disruptions, benign variants show near-zero deltas.

A **disruption attribution** (z-scored deltas vs the population of 184K ClinVar variants) \
identifies which features this variant disrupts most unusually. A z-score of -5σ on \
"helix structure" means this variant's helix disruption is 5 standard deviations beyond \
the population mean. Near-zero z-scores are noise.

Provide a clinical interpretation:
- Lead with the disruption story: which features were disrupted (large Δ) and which were preserved
- Use the gene function context to explain WHY disruption of those features matters for this gene
- Use the attribution to explain WHY the model predicts what it does
- When z-scores are high but deltas are small (<0.02), say so plainly: "statistically unusual but small in magnitude"
- Be skeptical of high z-scores on chromatin/epigenomic features for coding variants: these often reflect indirect correlations, not causal mechanisms
- Do NOT reference clinical prediction tools (CADD, AlphaMissense, SIFT, PolyPhen, etc.)
- 3-5 sentences for summary, 1 sentence for mechanism

Writing style:
- Write like a senior geneticist, not an AI. Be direct and concise.
- Never use em dashes. Use commas, periods, or parentheses instead.
- Never use "notably", "critically", "importantly", "striking", "painting a picture", "rewires"
- Never use "fully concordant", "fully consistent". Just say "consistent" or "matches".
- Don't use superlatives or hedging language. State facts plainly.
- Don't repeat the variant ID or coordinates in the summary (the user can see them).
- Specific numbers are good. Vague qualifiers ("profound", "massive", "broad") are not."""


def build_prompt(v: dict) -> str:
    """Build the interpretation prompt from a per-variant JSON."""
    score = v["score"]
    csq = v.get("consequence", "?")
    sub = v.get("substitution")
    lines = [f"## {v['gene']} — {v['id']}", f"Consequence: {csq}" + (f" ({sub})" if sub else "")]

    # Only show HGVS if consistent with consequence
    hgvsp = v.get("hgvsp") or ""
    hgvsc = v.get("hgvsc") or ""
    hgvs_consistent = not (
        ("=" in hgvsp and csq == "missense_variant")
        or ("=" not in hgvsp and hgvsp and csq == "synonymous_variant")
    )
    if hgvs_consistent and (hgvsp or hgvsc):
        parts = []
        if hgvsp:
            parts.append(f"Protein: {hgvsp}")
        if hgvsc:
            parts.append(f"Coding: {hgvsc}")
        lines.append("  |  ".join(parts))

    lines.append(f"**Predicted pathogenicity: {score * 100:.0f}%**")
    cal = calibration_text(score)
    if cal:
        lines.append(cal)
    lines.append("")

    # Curated disruption profile (filtered by quality + no tissue-specific/conservation/clinical)
    disruption = v.get("disruption", {})
    effect = v.get("effect", {})
    gt = v.get("gt", {})

    # Get curated head sets
    curated_dis = curated_group(set(disruption.keys()), quality_file=Path("head_quality.json"))
    curated_dis_heads = set()
    for heads in curated_dis.values():
        curated_dis_heads.update(heads)

    # Head stats (mean/std) come from builds/heads.json — each head has mean/std fields
    import json as _json
    _heads_path = Path("builds/heads.json")
    _heads_data = _json.loads(_heads_path.read_text()) if _heads_path.exists() else {}
    _head_stats = {h: {"mean": info["mean"], "std": info["std"]}
                   for h, info in _heads_data.get("heads", {}).items()
                   if "mean" in info and "std" in info}

    if disruption and curated_dis_heads:
        # Filter, compute z-scores, sort by |z|
        filtered = []
        for name, val in disruption.items():
            if name not in curated_dis_heads:
                continue
            if isinstance(val, list) and len(val) == 2:
                ref_val, var_val = val
                delta = var_val - ref_val
            elif isinstance(val, (int, float)):
                delta = val
                ref_val = var_val = None
            else:
                continue
            s = _head_stats.get(name, {})
            z = (delta - s.get("mean", 0)) / s["std"] if s.get("std", 0) > 0 else 0.0
            filtered.append((name, ref_val, var_val, delta, z))
        filtered.sort(key=lambda x: abs(x[4]), reverse=True)

        lines.append(f"### Disruption Profile ({len(filtered)} heads, ranked by |z-score|)")
        lines.append("Features with |z| > 2σ are nominally significant (p < 0.05).")
        lines.append("| Feature | ref | var | delta | z-score | Database |")
        lines.append("|---------|-----|-----|-------|---------|----------|")
        for name, ref_val, var_val, delta, z in filtered:
            ref_str = f"{ref_val:.3f}" if ref_val is not None else ""
            var_str = f"{var_val:.3f}" if var_val is not None else ""
            gt_val = gt.get(name)
            gt_str = f"{gt_val:.3f}" if isinstance(gt_val, (int, float)) else ""
            sig = " **" if abs(z) >= 2 else ""
            lines.append(f"| {display_name(name)} | {ref_str} | {var_str} | {delta:+.3f} | {z:+.1f}\u03c3{sig} | {gt_str} |")
        lines.append("")

    if effect:
        # Filter out clinical predictors from effect heads
        filtered_eff = [(name, val) for name, val in effect.items()
                        if name not in _EFFECT_PREDICTOR_HEADS and isinstance(val, (int, float))]
        filtered_eff.sort(key=lambda x: abs(x[1]), reverse=True)
        if filtered_eff:
            lines.append(f"### Effect Predictions ({len(filtered_eff)} heads, ranked by |score|)")
            lines.append("| Feature | score | Database |")
            lines.append("|---------|-------|----------|")
            for name, val in filtered_eff:
                gt_val = gt.get(name)
                gt_str = f"{gt_val:.3f}" if isinstance(gt_val, (int, float)) else ""
                lines.append(f"| {display_name(name)} | {val:.3f} | {gt_str} |")
            lines.append("")

    # Additional context
    context_parts = []
    if v.get("impact"):
        context_parts.append(f"VEP impact: {v['impact']}")
    if v.get("exon"):
        context_parts.append(f"Exon: {v['exon']}")
    if v.get("domains"):
        dom_str = v["domains"] if isinstance(v["domains"], str) else ", ".join(
            d.get("name") or d.get("id", "?") for d in v["domains"][:3]
        )
        context_parts.append(f"Domains: {dom_str}")
    if v.get("loeuf") is not None:
        context_parts.append(f"LOEUF: {v['loeuf']:.3f}")
    if v.get("gnomad") is not None:
        context_parts.append(f"gnomAD AF: {v['gnomad']:.2e}")
    if context_parts:
        lines.append("### Additional Context")
        lines.extend(f"- {p}" for p in context_parts)

    return "\n".join(lines)
