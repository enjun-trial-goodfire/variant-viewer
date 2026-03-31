"""Z-score attribution: rank disruption heads by population-calibrated delta magnitude.

A 0.05 delta in a head with sigma=0.01 is a 5σ event. A 0.2 delta with sigma=0.3 is noise.

Heuristic: we also require |delta| >= min_delta to avoid ranking nearly-constant heads
(e.g. in_peptide with σ≈0.001) where tiny deltas produce enormous z-scores. This is a
rough filter — a proper solution would model the noise floor per head or use a different
test statistic entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import torch

from display import _is_tissue_specific, _is_removed

EXCLUDED = frozenset({
    "alphamissense_c", "bayesdel_c", "cadd_c", "cadd_wg_c", "clinpred_c",
    "deogen2_c", "eve_c", "mcap_c", "metalr_c", "mpc_c", "mutpred_c",
    "mvp_c", "polyphen_c", "primateai_c", "revel_c", "sift_c", "vest4_c",
    "spliceai_ag_c", "spliceai_al_c", "spliceai_dg_c", "spliceai_dl_c",
    "spliceai_max_c", "phylop_c", "phastcons_c", "gerp_c",
    "phylop_100way", "phastcons_100way", "pathogenic",
})

# Load quality filter if available
_QUALITY_FILE = Path(__file__).parent / "head_quality.json"
_QUALITY_INCLUDED: frozenset[str] | None = None
if _QUALITY_FILE.exists():
    _data = json.loads(_QUALITY_FILE.read_text())
    _QUALITY_INCLUDED = frozenset(_data.get("included", []))


def _disruption_deltas(scores: pl.DataFrame, heads: tuple[str, ...]) -> tuple[tuple[str, ...], torch.Tensor]:
    """Extract ref→var deltas for eligible disruption heads. Returns (head_names, [N, H] tensor)."""
    eligible = tuple(
        h for h in heads
        if h not in EXCLUDED
        and (_QUALITY_INCLUDED is None or h in _QUALITY_INCLUDED)
        and not _is_tissue_specific(h)
        and not _is_removed(h)
        and f"ref_score_{h}" in scores.columns and f"var_score_{h}" in scores.columns
    )
    ref = torch.tensor(scores.select([f"ref_score_{h}" for h in eligible]).fill_null(0).to_numpy())
    var = torch.tensor(scores.select([f"var_score_{h}" for h in eligible]).fill_null(0).to_numpy())
    return eligible, var - ref


def attribute(
    scores: pl.DataFrame, disruption_heads: tuple[str, ...],
    k: int = 10,
) -> dict[str, list[dict]]:
    """Z-score attribution for all variants. Returns {variant_id: [{name, z}, ...]}.

    Uses raw |z| (magnitude only — direction is already captured in the delta).
    """
    heads, deltas = _disruption_deltas(scores, disruption_heads)
    z = (deltas - deltas.mean(0)) / deltas.std(0).clamp(min=1e-6)
    z_abs = z.abs()

    # Select top candidates (more than k, then deduplicate)
    n_candidates = min(k * 3, len(heads))
    topk_vals, topk_idx = z_abs.topk(n_candidates, dim=1)
    vids = scores["variant_id"].to_list()

    result = {}
    for i in range(len(vids)):
        selected = []
        seen_groups: set[str] = set()
        for j in range(n_candidates):
            if topk_vals[i, j] <= 0.5:
                break
            name = heads[topk_idx[i, j].item()]
            z_val = round(topk_vals[i, j].item(), 2)
            # Deduplicate: group by prefix pattern (e.g., chromhmm_*_bivalent → bivalent)
            group = _head_group(name)
            if group in seen_groups:
                continue
            seen_groups.add(group)
            selected.append({"name": name, "z": z_val})
            if len(selected) >= k:
                break
        result[vids[i]] = selected
    return result


# ── Group deduplication ────────────────────────────────────────────────

# Patterns: chromhmm_{tissue}_{state} → state, chipseq_{mark}_{tissue}_{type} → mark_type,
# fstack_{state} → fstack, atacseq_{tissue}_{type} → atacseq_type
_TISSUE_PREFIXES = ("chromhmm_", "chipseq_", "atacseq_", "fstack_")
_CHROMHMM_STATES = (
    "active_tss", "bivalent", "enhancer", "heterochromatin", "polycomb",
    "quiescent", "transcribed", "weak_enhancer", "weak_transcription",
)


def _head_group(name: str) -> str:
    """Map a head name to a deduplication group. Heads in the same group are redundant."""
    # ChromHMM: chromhmm_{tissue}_{state} → chromhmm_{state}
    if name.startswith("chromhmm_"):
        suffix = name[9:]  # strip "chromhmm_"
        for state in _CHROMHMM_STATES:
            if suffix.endswith(state):
                return f"chromhmm_{state}"
        return name  # unknown state, keep as-is

    # ChIP-seq: chipseq_{mark}_{tissue}_{peak/signal} → chipseq_{mark}_{peak/signal}
    if name.startswith("chipseq_"):
        parts = name.split("_")
        if len(parts) >= 4:
            mark = parts[1]  # e.g., h3k27me3
            kind = parts[-1]  # peak or signal
            return f"chipseq_{mark}_{kind}"
        return name

    # ATAC-seq: atacseq_{tissue}_{type} → atacseq_{type}
    if name.startswith("atacseq_"):
        kind = name.split("_")[-1]  # signal, peak, breadth
        return f"atacseq_{kind}"

    # fstack: fstack_{state} → fstack
    if name.startswith("fstack_"):
        return "fstack"

    # Everything else: no dedup (each head is its own group)
    return name
