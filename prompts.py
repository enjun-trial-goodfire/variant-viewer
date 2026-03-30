"""Claude interpretation prompts for the variant viewer.

Single source of truth for the system prompt and prompt builder.
Used by serve.py (on-demand) and pipeline/interpret.py (batch).

Schema contract: all string fields are "" (never null), all lists are [],
all sparse dicts (disruption/effect/gt) use missing key = 0.
Only loeuf, gnomad, allele_id, n_submissions, last_evaluated are nullable.
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
- **delta**: var - ref = the disruption signal

Large negative deltas mean the variant disrupts that feature. Near-zero deltas mean \
the feature is preserved. This is the key signal: pathogenic variants show large \
disruptions, benign variants show near-zero deltas.

A separate **diff probe** predicts what existing clinical tools (CADD, AlphaMissense, \
REVEL, SpliceAI, etc.) would score for this variant. These are NOT external lookups. \
They are Evo2's internal predictions of what those tools would say, based purely on DNA \
sequence context.

A **disruption attribution** (z-scored deltas vs the population of 184K ClinVar variants) \
identifies which features this variant disrupts most unusually. A z-score of 5 on \
"helix structure" means this variant's helix disruption is 5 standard deviations beyond \
the population mean. Near-zero z-scores are noise.

Provide a clinical interpretation:
- Lead with the disruption story: which features were disrupted (large delta) and preserved
- Integrate the effect predictions: what do CADD, AlphaMissense, REVEL, Pfam, etc. predict?
  Compare with database values when available. Strong agreement = high confidence.
- Use the attribution to explain WHY the model predicts what it does
- Note any disagreements between pathogenicity and predicted clinical scores
- Nearest neighbor consensus is strong independent evidence
- When z-scores are high but deltas are small (<0.02), say so: "statistically unusual but small in magnitude"
- Be skeptical of chromatin/epigenomic z-scores for coding variants: often indirect correlations
- 3-5 sentences for summary, 1 sentence for mechanism

Writing style:
- Write like a senior geneticist, not an AI. Be direct and concise.
- Never use em dashes. Use commas, periods, or parentheses instead.
- Never use "notably", "critically", "importantly", "striking", "painting a picture", "rewires"
- Never use "fully concordant", "fully consistent". Just say "consistent" or "matches".
- No superlatives or hedging. State facts plainly.
- Don't repeat the variant ID or coordinates (the user can see them).
- Specific numbers are good. Vague qualifiers ("profound", "massive", "broad") are not."""


def _fmt(val, fmt=".3f") -> str:
    """Format a numeric value, empty string if None."""
    return f"{val:{fmt}}" if val is not None else ""


def _signal_quality(delta: float) -> str:
    """Classify disruption signal strength from delta magnitude."""
    d = abs(delta)
    return "strong" if d > 0.05 else "moderate" if d > 0.02 else "weak (small delta)"


def _agreement(prediction: float, database: float) -> str:
    """Classify prediction vs database agreement."""
    err = abs(prediction - database)
    return "good" if err < 0.1 else "fair" if err < 0.25 else "poor"


def build_prompt(v: dict) -> str:
    """Build the interpretation prompt from a per-variant JSON.

    Assumes strict schema: strings are "", lists are [], sparse dicts use absence = 0.
    Only loeuf/gnomad are nullable.
    """
    disruption = v["disruption"]
    effect = v["effect"]
    gt = v["gt"]
    attribution = v["attribution"]
    neighbors = v["neighbors"]

    lines = [
        f"## {v['gene']} — {v['id']}",
        f"Consequence: {v['consequence']}" + (f" ({v['substitution']})" if v["substitution"] else ""),
    ]

    # HGVS (only if consistent with consequence)
    hgvsp, hgvsc, csq = v["hgvsp"], v["hgvsc"], v["consequence"]
    synonymous_mismatch = "=" in hgvsp and csq == "missense_variant"
    missense_mismatch = "=" not in hgvsp and hgvsp and csq == "synonymous_variant"
    if not synonymous_mismatch and not missense_mismatch:
        hgvs = "  |  ".join(
            [f"Protein: {hgvsp}" for _ in [1] if hgvsp] + [f"Coding: {hgvsc}" for _ in [1] if hgvsc]
        )
        if hgvs:
            lines.append(hgvs)

    # Label
    label = v["label"]
    lines.append(
        "ClinVar: VUS (Variant of Uncertain Significance)" if label == "VUS"
        else f"ClinVar: {label} ({v['significance']}), {v['stars']} stars"
    )
    if v["disease"]:
        lines.append(f"Disease: {v['disease']}")

    lines.append(f"**Predicted pathogenicity: {v['score'] * 100:.0f}%**")
    cal = calibration_text(v["score"])
    if cal:
        lines.append(cal)
    lines.append("")

    # Disruption attribution
    if attribution:
        lines.append("### Disruption Attribution (ranked by gated z-score)")
        lines.append("| Feature | z-score | delta | Database | Signal quality |")
        lines.append("|---------|---------|-------|----------|---------------|")
        for h in attribution[:15]:
            name = h["name"]
            delta = disruption.get(name, 0)
            db = gt.get(name)
            lines.append(
                f"| {display_name(name)} | {h.get('z', 0):.1f}\u03c3"
                f" | {delta:+.3f} | {_fmt(db)} | {_signal_quality(delta)} |"
            )
        lines.append("")

    # Effect predictions
    if effect:
        ranked = sorted(effect.items(), key=lambda kv: -abs(kv[1] - 0.5))[:15]
        lines.append("### Variant Effect Predictions (from diff view, ranked by decisiveness)")
        lines.append("| Feature | Prediction | Database | Agreement |")
        lines.append("|---------|-----------|----------|-----------|")
        for name, val in ranked:
            db = gt.get(name)
            agreement = _agreement(val, db) if db is not None else ""
            lines.append(f"| {display_name(name)} | {val:.3f} | {_fmt(db)} | {agreement} |")
        lines.append("")

    # Neighbors
    if neighbors:
        lines.append(f"### Nearest Neighbors: {v['nP']} pathogenic, {v['nB']} benign, {v['nV']} VUS")
        for nb in neighbors[:5]:
            lines.append(f"- {nb['gene']} ({nb['label']}, pathogenicity={nb['score'] * 100:.0f}%, similarity={nb.get('similarity', 0) * 100:.0f}%)")
        lines.append("")

    # Context (genuinely nullable fields checked here)
    context = [
        f"VEP impact: {v['impact']}" if v["impact"] else None,
        f"Exon: {v['exon']}" if v["exon"] else None,
        f"Domains: {', '.join(d.get('name') or d.get('id', '?') for d in v['domains'][:3])}" if v["domains"] else None,
        f"LOEUF: {v['loeuf']:.3f}" if v["loeuf"] is not None else None,
        f"gnomAD AF: {v['gnomad']:.2e}" if v["gnomad"] is not None else None,
    ]
    context = [c for c in context if c]
    if context:
        lines.append("### Additional Context")
        lines.extend(f"- {c}" for c in context)

    return "\n".join(lines)
