"""Modular feature-group definitions for Phase 3 pairwise comparisons.

Groups are discovered from actual column names in ``variants_wide.parquet`` so the
same code works across schema tweaks. Revise token lists below if annotation naming changes.

Reused patterns (from typical DuckDB export):
  * ``z_*`` — disruption z-scores
  * ``ref_*`` / ``var_*`` — raw ref/alt projection tracks (excludes ``ref_pred_*`` / ``var_pred_*``)
  * ``dist_*`` / ``spread_*`` — distance/spread to peaks (paired with ref/var tracks)
  * ``w0_*``, ``w2_*``, ``w64_*`` — multi-scale windows
  * ``ref_pred_*`` / ``var_pred_*`` — coarse structural / transcript layout predictors
"""

from __future__ import annotations

from typing import Iterable

# --- Subgroup keyword rules (substring match on lowercased column name) ---

_REGULATORY = (
    "atacseq",
    "chipseq",
    "chromhmm",
    "cpg",
    "fstack",
)
_SPLICE = (
    "splice",
    "branchpoint",
    "polypyrimidine",
    "exon_to_intron",
    "intron_to_exon",
)
_PTM = ("ptm_",)
_INTERPRO = ("interpro",)
_AMINO = ("amino_acid",)
_STRUCTURE = (
    "secondary_structure",
    "disordered",
    "plddt",
    "ppi_interface",
    "linear_interacting",
    "region_",
    "is_start_codon",
    "is_ptc",
    "in_chain",
    "in_interpro_domain",
)


def _lower(c: str) -> str:
    return c.lower()


def _is_raw_ref_var(c: str) -> bool:
    return (c.startswith("ref_") or c.startswith("var_")) and not (
        c.startswith("ref_pred_") or c.startswith("var_pred_")
    )


def _is_structure_pred(c: str) -> bool:
    return c.startswith("ref_pred_") or c.startswith("var_pred_")


def _classify_z_subgroup(c: str) -> str:
    z = _lower(c)
    if any(t in z for t in _REGULATORY):
        return "chromatin_regulatory"
    if any(t in z for t in _SPLICE):
        return "splice"
    if any(t in z for t in _PTM):
        return "ptm"
    if any(t in z for t in _INTERPRO):
        return "interpro_domain"
    if any(t in z for t in _AMINO):
        return "amino_acid"
    if any(t in z for t in _STRUCTURE):
        return "structure_context"
    return "other"


def _classify_refvar_subgroup(c: str) -> str:
    z = _lower(c)
    if any(t in z for t in _REGULATORY):
        return "chromatin_regulatory"
    if any(t in z for t in _SPLICE):
        return "splice"
    if any(t in z for t in _PTM):
        return "ptm"
    if any(t in z for t in _INTERPRO):
        return "interpro_domain"
    if any(t in z for t in _AMINO):
        return "amino_acid"
    if any(t in z for t in _STRUCTURE):
        return "structure_context"
    return "other"


def _classify_dist_spread_subgroup(c: str) -> str:
    z = _lower(c)
    if any(t in z for t in _REGULATORY):
        return "chromatin_regulatory"
    if any(t in z for t in _SPLICE):
        return "splice"
    if any(t in z for t in _PTM):
        return "ptm"
    if any(t in z for t in _INTERPRO):
        return "interpro_domain"
    if any(t in z for t in _AMINO):
        return "amino_acid"
    if any(t in z for t in _STRUCTURE):
        return "structure_context"
    return "other"


def discover_feature_groups(columns: Iterable[str]) -> dict[str, list[str]]:
    """Map group name -> sorted list of numeric column names present in *columns*."""
    s = set(columns)
    out: dict[str, list[str]] = {}

    def add(name: str, members: list[str]) -> None:
        m = sorted(set(members) & s)
        if m:
            out[name] = m

    z_cols = [c for c in s if c.startswith("z_")]
    add("disruption_z_all", z_cols)
    for sub, pred in (
        ("disruption_z_chromatin_regulatory", lambda c: _classify_z_subgroup(c) == "chromatin_regulatory"),
        ("disruption_z_splice", lambda c: _classify_z_subgroup(c) == "splice"),
        ("disruption_z_ptm", lambda c: _classify_z_subgroup(c) == "ptm"),
        ("disruption_z_interpro_domain", lambda c: _classify_z_subgroup(c) == "interpro_domain"),
        ("disruption_z_amino_acid", lambda c: _classify_z_subgroup(c) == "amino_acid"),
        ("disruption_z_structure_context", lambda c: _classify_z_subgroup(c) == "structure_context"),
        ("disruption_z_other", lambda c: _classify_z_subgroup(c) == "other"),
    ):
        add(sub, [c for c in z_cols if pred(c)])

    rv = [c for c in s if _is_raw_ref_var(c)]
    add("annotation_ref_var_all", rv)
    for sub, key in (
        ("annotation_ref_var_chromatin_regulatory", "chromatin_regulatory"),
        ("annotation_ref_var_splice", "splice"),
        ("annotation_ref_var_ptm", "ptm"),
        ("annotation_ref_var_interpro_domain", "interpro_domain"),
        ("annotation_ref_var_amino_acid", "amino_acid"),
        ("annotation_ref_var_structure_context", "structure_context"),
        ("annotation_ref_var_other", "other"),
    ):
        add(sub, [c for c in rv if _classify_refvar_subgroup(c) == key])

    sp = [c for c in s if c.startswith("ref_pred_") or c.startswith("var_pred_")]
    add("structure_prediction_ref_var", sp)

    ds = [c for c in s if c.startswith("dist_") or c.startswith("spread_")]
    add("distance_spread_all", ds)
    for sub, key in (
        ("distance_spread_chromatin_regulatory", "chromatin_regulatory"),
        ("distance_spread_splice", "splice"),
        ("distance_spread_ptm", "ptm"),
        ("distance_spread_interpro_domain", "interpro_domain"),
        ("distance_spread_amino_acid", "amino_acid"),
        ("distance_spread_structure_context", "structure_context"),
        ("distance_spread_other", "other"),
    ):
        add(sub, [c for c in ds if _classify_dist_spread_subgroup(c) == key])

    add("multiscale_w0_all", [c for c in s if c.startswith("w0_")])
    add("multiscale_w2_all", [c for c in s if c.startswith("w2_")])
    add("multiscale_w64_all", [c for c in s if c.startswith("w64_")])

    return out


def summary_group_order(groups: dict[str, list[str]]) -> list[str]:
    """Stable ordering for reporting: coarse blocks first, then alphabetically within."""
    priority = [
        "disruption_z_all",
        "disruption_z_chromatin_regulatory",
        "disruption_z_splice",
        "disruption_z_ptm",
        "disruption_z_interpro_domain",
        "disruption_z_amino_acid",
        "disruption_z_structure_context",
        "disruption_z_other",
        "annotation_ref_var_all",
        "annotation_ref_var_chromatin_regulatory",
        "annotation_ref_var_splice",
        "annotation_ref_var_ptm",
        "annotation_ref_var_interpro_domain",
        "annotation_ref_var_amino_acid",
        "annotation_ref_var_structure_context",
        "structure_prediction_ref_var",
        "distance_spread_all",
        "multiscale_w0_all",
        "multiscale_w2_all",
        "multiscale_w64_all",
    ]
    ordered = [g for g in priority if g in groups]
    rest = sorted(g for g in groups if g not in ordered)
    return ordered + rest
