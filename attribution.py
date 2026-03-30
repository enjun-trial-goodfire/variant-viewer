"""Z-score attribution: rank disruption heads by population-calibrated delta magnitude.

A 0.05 delta in a head with sigma=0.01 is a 5σ event. A 0.2 delta with sigma=0.3 is noise.

Heuristic: we also require |delta| >= min_delta to avoid ranking nearly-constant heads
(e.g. in_peptide with σ≈0.001) where tiny deltas produce enormous z-scores. This is a
rough filter — a proper solution would model the noise floor per head or use a different
test statistic entirely.
"""

from __future__ import annotations

import polars as pl
import torch

EXCLUDED = frozenset({
    "alphamissense_c", "bayesdel_c", "cadd_c", "cadd_wg_c", "clinpred_c",
    "deogen2_c", "eve_c", "mcap_c", "metalr_c", "mpc_c", "mutpred_c",
    "mvp_c", "polyphen_c", "primateai_c", "revel_c", "sift_c", "vest4_c",
    "spliceai_ag_c", "spliceai_al_c", "spliceai_dg_c", "spliceai_dl_c",
    "spliceai_max_c", "phylop_c", "phastcons_c", "gerp_c",
    "phylop_100way", "phastcons_100way", "pathogenic",
})


def _disruption_deltas(scores: pl.DataFrame, heads: tuple[str, ...]) -> tuple[tuple[str, ...], torch.Tensor]:
    """Extract ref→var deltas for eligible disruption heads. Returns (head_names, [N, H] tensor)."""
    eligible = tuple(
        h for h in heads
        if h not in EXCLUDED and f"ref_score_{h}" in scores.columns and f"var_score_{h}" in scores.columns
    )
    ref = torch.tensor(scores.select([f"ref_score_{h}" for h in eligible]).fill_null(0).to_numpy())
    var = torch.tensor(scores.select([f"var_score_{h}" for h in eligible]).fill_null(0).to_numpy())
    return eligible, var - ref


def attribute(
    scores: pl.DataFrame, disruption_heads: tuple[str, ...],
    k: int = 10, delta_midpoint: float = 0.03, delta_temperature: float = 0.01,
) -> dict[str, list[dict]]:
    """Z-score attribution for all variants. Returns {variant_id: [{name, z}, ...]}.

    Uses |z| (magnitude only — direction is already captured in the delta).
    Applies sigmoid gating on |delta| to smoothly downweight heads with tiny deltas,
    avoiding spurious high z-scores from near-constant heads. The gate is:
        weight = sigmoid((|delta| - midpoint) / temperature)
    so deltas >> midpoint get weight ≈ 1, deltas << midpoint get weight ≈ 0.
    """
    heads, deltas = _disruption_deltas(scores, disruption_heads)
    z = (deltas - deltas.mean(0)) / deltas.std(0).clamp(min=1e-6)
    z_abs = z.abs()

    # Sigmoid gate: smoothly downweight z-scores for heads with tiny deltas
    gate = torch.sigmoid((deltas.abs() - delta_midpoint) / delta_temperature)
    z_gated = z_abs * gate

    topk_vals, topk_idx = z_gated.topk(min(k, len(heads)), dim=1)
    vids = scores["variant_id"].to_list()

    return {
        vids[i]: [
            {"name": heads[topk_idx[i, j].item()], "z": round(topk_vals[i, j].item(), 2)}
            for j in range(topk_vals.shape[1])
            if topk_vals[i, j] > 0.5
        ]
        for i in range(len(vids))
    }
