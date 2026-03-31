"""Shared constants for the variant viewer.

Single source of truth for biology constants, calibration data, probe config,
and model settings. Imported by build.py, serve.py, and pipeline scripts.
"""

PROBE_NAME = "probe_v11"
CLAUDE_MODEL = "claude-sonnet-4-6"
EVAL_KEYS = (("correlation", "r"), ("auc", "AUC"), ("accuracy", "acc"))

# Consequence → integer encoding (deterministic order by frequency in deconfounded-full)
CONSEQUENCE_CLASSES = (
    "missense_variant", "intron_variant", "synonymous_variant", "nonsense",
    "frameshift_variant", "non-coding_transcript_variant", "splice_donor_variant",
    "splice_acceptor_variant", "5_prime_UTR_variant", "3_prime_UTR_variant",
    "splice_region_variant", "start_lost", "inframe_deletion", "inframe_insertion",
    "inframe_indel", "stop_lost", "genic_downstream_transcript_variant",
    "genic_upstream_transcript_variant", "no_sequence_alteration",
    "initiator_codon_variant",
)

_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SWAP_CLASSES = tuple(f"{a}>{b}" for a in _AA for b in _AA if a != b)

LABEL_TO_IDX = {"benign": 0, "pathogenic": 1, "VUS": 2}

# Calibration: % of variants that are actually pathogenic in each score bin.
# Source: labeled variants from clinvar_evo2_deconfounded_full.
# Bounds are [lo, hi) — the last bin uses 1.01 to include score == 1.0.
CALIBRATION = {
    (0.0, 0.1): 1.8, (0.1, 0.3): 15.6, (0.3, 0.5): 44.4,
    (0.5, 0.7): 69.6, (0.7, 0.9): 88.1, (0.9, 1.01): 96.7,
}


def calibration_text(score: float) -> str:
    """Human-readable calibration context for a pathogenicity score."""
    for (lo, hi), rate in CALIBRATION.items():
        if lo <= score < hi:
            return f"Calibration: among labeled variants scoring {lo*100:.0f}\u2013{hi*100:.0f}%, {rate:.0f}% are pathogenic."
    return ""
