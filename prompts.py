"""Claude interpretation prompts for the variant viewer.

Single source of truth for the system prompt and prompt builder.
Used by serve.py (on-demand) and pipeline/interpret.py (batch).
"""

from constants import calibration_text
from display import display_name

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

    # Label context
    label = v.get("label", "?")
    if label != "VUS":
        lines.append(f"ClinVar: {label} ({v.get('significance', '')}), {v.get('stars', '?')} stars")
    else:
        lines.append("ClinVar: VUS (Variant of Uncertain Significance)")
    if v.get("disease"):
        lines.append(f"Disease: {v['disease']}")

    lines.append(f"**Predicted pathogenicity: {score * 100:.0f}%**")
    cal = calibration_text(score)
    if cal:
        lines.append(cal)
    lines.append("")

    # Attribution heads — normalize both schemas to a flat list with {name, kind, weight}
    attr = v.get("attribution") or {}
    disruption = v.get("disruption", {})
    effect = v.get("effect", {})
    gt = v.get("gt", {})

    # Schema v2: {"heads": [{name, kind, coefficient}, ...]}
    # Schema v1: {"effect": [{name, score, contribution}, ...], "disruption": [...]}
    if "heads" in attr:
        attr_heads = attr["heads"]
        weight_key = "coefficient"
    elif "effect" in attr or "disruption" in attr:
        attr_heads = [
            {**h, "kind": "effect", "coefficient": h.get("contribution", 0)}
            for h in attr.get("effect", [])
        ] + [
            {**h, "kind": "disruption", "coefficient": h.get("contribution", 0)}
            for h in attr.get("disruption", [])
        ]
        weight_key = "coefficient"
    else:
        attr_heads = []
        weight_key = "coefficient"

    top = sorted(attr_heads, key=lambda h: abs(h.get(weight_key, 0)), reverse=True)[:15]
    if top:
        lines.append("### Attribution (features driving the prediction, ranked by importance)")
        lines.append("| Feature | Type | Weight | Value | Ref\u2192Var (\u0394) | Database |")
        lines.append("|---------|------|--------|-------|-------------|----------|")
        for h in top:
            name = h["name"]
            kind = h.get("kind", "?")
            coeff = h.get(weight_key, 0)
            val_str, delta_str = "\u2014", "\u2014"
            gt_val = gt.get(name)
            gt_str = f"{gt_val:.3f}" if isinstance(gt_val, (int, float)) else "\u2014"
            if kind == "disruption" and name in disruption:
                delta = disruption[name] if isinstance(disruption[name], (int, float)) else disruption[name][1] - disruption[name][0]
                val_str = f"\u0394={delta:+.3f}"
                delta_str = val_str
            elif kind == "effect" and name in effect:
                val_str = f"{effect[name]:.3f}"
            lines.append(f"| {display_name(name)} | {kind} | {coeff:+.4f} | {val_str} | {delta_str} | {gt_val} |")
        lines.append("")

    # Neighbors
    neighbors = v.get("neighbors", [])
    if neighbors:
        lines.append(f"### Nearest Neighbors: {v.get('nP', 0)} pathogenic, {v.get('nB', 0)} benign, {v.get('nV', 0)} VUS")
        for nb in neighbors[:5]:
            sim = nb.get("similarity", 0)
            lines.append(f"- {nb['gene']} ({nb['label']}, pathogenicity={nb['score'] * 100:.0f}%, similarity={sim * 100:.0f}%)")
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
