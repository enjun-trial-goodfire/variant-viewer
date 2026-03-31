"""Display names and grouping for probe heads.

Converts raw head keys (e.g., "phylop_100way", "pfam_PF00069") to
human-readable names. The logic is:
1. Check explicit overrides (for names that can't be derived)
2. Strip known prefixes (interpro_, pfam_, amino_acid_, chipseq_, etc.)
3. Apply formatting rules (acronyms uppercase, Pfam → domain lookup, etc.)
"""

# Names that can't be derived from the key — must be listed explicitly.
# If a name CAN be derived by stripping prefix + title-casing, don't add it here.
_OVERRIDES: dict[str, str] = {
    # Clinical predictors (acronyms/brand names)
    "cadd_c": "CADD", "cadd_wg_c": "CADD (whole-genome)",
    "revel_c": "REVEL", "alphamissense_c": "AlphaMissense",
    "sift_c": "SIFT", "polyphen_c": "PolyPhen-2", "eve_c": "EVE",
    "spliceai_max_c": "SpliceAI", "gnomad_af_c": "gnomAD AF",
    "spliceai_ag_c": "SpliceAI AG", "spliceai_al_c": "SpliceAI AL",
    "spliceai_dg_c": "SpliceAI DG", "spliceai_dl_c": "SpliceAI DL",
    "mpc_c": "MPC", "mvp_c": "MVP", "mcap_c": "M-CAP",
    "metalr_c": "MetaLR", "vest4_c": "VEST4", "primateai_c": "PrimateAI",
    "mutpred_c": "MutPred", "clinpred_c": "ClinPred", "deogen2_c": "DEOGEN2",
    "bayesdel_c": "BayesDel", "remm_c": "ReMM", "regulomedb_c": "RegulomeDB",
    # Conservation (proper names)
    "phylop_c": "PhyloP", "phylop_100way": "PhyloP 100-way",
    "phastcons_c": "PhastCons", "phastcons_100way": "PhastCons 100-way",
    "gerp_c": "GERP",
    # AlphaFold features (avoid confusion — "psi" is NOT splicing)
    "psi": "AF Psi Angle", "phi": "AF Phi Angle",
    "plddt": "AF Confidence (pLDDT)", "rsa": "Solvent Accessibility",
    "sasa": "Surface Area",
    # Irregular names that title-casing gets wrong
    "blosum62_c": "BLOSUM62", "loeuf_c": "LOEUF", "mw_c": "Mol. Weight",
    "gtex_max_tpm_c": "GTEx Max TPM", "gtex_n_tissues_c": "GTEx Tissues",
    "gtex_brain_max_c": "GTEx Brain Max",
    "aa_swap": "AA Substitution",
}

# Prefixes to strip when generating display names.
# Order matters: longer prefixes first to avoid partial matches.
_STRIP_PREFIXES = (
    "interpro_", "amino_acid_", "chipseq_", "atacseq_",
    "chromhmm_", "fstack_", "pfam_", "elm_", "gtex_", "ptm_",
    "secondary_structure_", "region_", "dna_shape_",
)

# First token → group category for auto_group()
_GROUP_MAP: dict[str, str] = {}
for _cat, _tokens in {
    "Conservation": ("phylop", "phastcons", "gerp"),
    "Protein": ("secondary", "disorder", "plddt", "rsa", "sasa", "phi", "psi", "ppi", "residue"),
    "Structure": ("in", "is", "has"),
    "InterPro": ("interpro",),
    "Pfam": ("pfam",),
    "ELM": ("elm",),
    "ChIP-seq": ("chipseq",),
    "ATAC-seq": ("atacseq",),
    "Chromatin": ("chromhmm", "fstack"),
    "Regulatory": ("ccre", "dna"),
    "Amino acid": ("amino",),
    "Sequence": ("codon", "trinuc", "gc", "cpg", "syn"),
    "Substitution": ("aa", "blosum62", "grantham", "hydrophobicity", "volume", "mw"),
    "PTM": ("ptm",),
    "Expression": ("gtex",),
    "Region": ("region", "exon", "n", "trf", "segdup", "recomb"),
    "Constraint": ("loeuf",),
    "Clinical": (
        "cadd", "revel", "alphamissense", "sift", "polyphen", "eve", "bayesdel",
        "clinpred", "deogen2", "mcap", "metalr", "mpc", "mutpred", "mvp",
        "primateai", "vest4", "remm", "regulomedb",
    ),
    "Splice": ("spliceai",),
    "Variant effect": ("splice", "charge", "consequence", "impact", "csq"),
    "Population": ("gnomad",),
    "Pathogenicity": ("pathogenic",),
}.items():
    _GROUP_MAP.update({t: _cat for t in _tokens})


def display_name(h: str, domain_cache: dict[str, str] | None = None) -> str:
    """Human-readable head name."""
    if h in _OVERRIDES:
        return _OVERRIDES[h]

    base = h.removesuffix("_c")
    for prefix in _STRIP_PREFIXES:
        if base.startswith(prefix):
            stripped = base[len(prefix):]
            # Single/two-char results keep the prefix (amino_acid_L → "Amino Acid L")
            if len(stripped) <= 2:
                return prefix.rstrip("_").replace("_", " ").title() + " " + stripped.upper()
            base = stripped
            break

    base = base.replace("_", " ")

    # Pfam accessions → look up domain name
    if base.startswith("PF"):
        name = (domain_cache or {}).get(f"Pfam:{base}")
        return f"{name} ({base})" if name else base

    return base.title()


def auto_group(heads: set[str]) -> dict[str, list[str]]:
    """Group heads into categories by first-token prefix."""
    groups: dict[str, list[str]] = {}
    for h in sorted(heads):
        cat = _GROUP_MAP.get(h.split("_")[0], "Other")
        groups.setdefault(cat, []).append(h)
    return groups


# ── Curated grouping (quality-filtered, no tissue-specific) ──────────

import json as _json
import re as _re
from collections import OrderedDict as _OrderedDict
from pathlib import Path as _Path

# Heads to exclude from annotation expansion (shown elsewhere in the UI)
_PREDICTOR_HEADS = frozenset({
    "phylop_100way", "phastcons_100way", "gerp_c",  # conservation → predictors card
    "cadd_wg_c", "remm_c", "regulomedb_c",          # clinical → predictors card
    "loeuf_c",                                        # constraint → inline gene name
})

# Effect heads shown in the Computational Predictors card (exclude from annotation expansion)
_EFFECT_PREDICTOR_HEADS = frozenset({
    "alphamissense_c", "bayesdel_c", "cadd_c", "clinpred_c", "deogen2_c",
    "eve_c", "mcap_c", "metalr_c", "mpc_c", "mutpred_c", "mvp_c",
    "polyphen_c", "primateai_c", "revel_c", "sift_c", "vest4_c",
    "gnomad_af_c", "pathogenic",
    "spliceai_ag_c", "spliceai_al_c", "spliceai_dg_c", "spliceai_dl_c", "spliceai_max_c",
})

# Categories to remove entirely
_REMOVED_PREFIXES = frozenset({"codon", "trinuc", "syn", "gtex"})
_REMOVED_EXACT = frozenset({"gc_content"})

# Tissue-specific pattern: chipseq/atacseq/chromhmm with tissue name (not breadth/count)
_TISSUE_PATTERN = _re.compile(
    r"^(chipseq|atacseq|chromhmm)_.*"
    r"(?<!_breadth)(?<!_count)$"
)

# Curated category mapping (ordered for display)
_CURATED_MAP = _OrderedDict({
    "Protein": ("secondary", "disorder", "plddt", "rsa", "sasa", "phi", "psi", "ppi", "residue"),
    "Structure": ("in", "is", "has"),
    "Regulatory": ("atacseq", "chipseq", "chromhmm", "fstack", "ccre", "dna"),
    "Domains": ("interpro",),
    "ELM Motifs": ("elm",),
    "Amino Acid": ("amino",),
    "PTM": ("ptm",),
    "Region": ("region", "exon", "n", "trf", "segdup", "recomb"),
})


def _is_tissue_specific(h: str) -> bool:
    """True if head is a tissue-specific epigenomic measurement (not a summary)."""
    if h.endswith("_breadth") or h.endswith("_count"):
        return False
    if h.startswith("fstack_"):
        return False  # fstack are summary states, not tissue-specific
    return bool(_TISSUE_PATTERN.match(h))


def _is_removed(h: str) -> bool:
    """True if head belongs to a removed category."""
    if h in _REMOVED_EXACT or h in _PREDICTOR_HEADS:
        return True
    first_token = h.split("_")[0]
    return first_token in _REMOVED_PREFIXES


def curated_group(
    heads: set[str],
    quality_file: _Path | None = None,
) -> dict[str, list[str]]:
    """Group heads into curated categories, filtered by quality and excluding tissue-specific.

    Args:
        heads: Set of all head keys to group.
        quality_file: Path to head_quality.json. If provided, only include passing heads.

    Returns:
        Ordered dict of {category_name: [head_key, ...]} with only curated categories.
    """
    # Load quality filter
    quality_included: set[str] | None = None
    if quality_file and quality_file.exists():
        data = _json.loads(quality_file.read_text())
        quality_included = set(data.get("included", []))

    # Build token → category lookup
    token_to_cat: dict[str, str] = {}
    for cat, tokens in _CURATED_MAP.items():
        for t in tokens:
            token_to_cat[t] = cat

    groups: dict[str, list[str]] = _OrderedDict((cat, []) for cat in _CURATED_MAP)

    for h in sorted(heads):
        # Skip removed categories
        if _is_removed(h):
            continue
        # Skip tissue-specific
        if _is_tissue_specific(h):
            continue
        # Skip if doesn't pass quality (when filter available)
        if quality_included is not None and h not in quality_included:
            continue
        # Assign category
        first_token = h.split("_")[0]
        cat = token_to_cat.get(first_token)
        if cat:
            groups[cat].append(h)

    # Remove empty categories
    return _OrderedDict((cat, heads) for cat, heads in groups.items() if heads)


def curated_effect_group(heads: set[str]) -> dict[str, list[str]]:
    """Group effect heads, excluding clinical predictors (shown in Predictors card).

    Returns ordered dict: {category: [head_keys]}.
    """
    _EFFECT_MAP = _OrderedDict({
        "Substitution": ("aa", "blosum62", "grantham", "hydrophobicity", "volume", "mw"),
        "Variant Effects": ("splice", "charge", "consequence", "impact", "csq"),
        "Pfam Domains": ("pfam",),
    })

    token_to_cat: dict[str, str] = {}
    for cat, tokens in _EFFECT_MAP.items():
        for t in tokens:
            token_to_cat[t] = cat

    groups: dict[str, list[str]] = _OrderedDict((cat, []) for cat in _EFFECT_MAP)

    for h in sorted(heads):
        if h in _EFFECT_PREDICTOR_HEADS:
            continue
        first_token = h.split("_")[0]
        cat = token_to_cat.get(first_token)
        if cat:
            groups[cat].append(h)

    return _OrderedDict((cat, hs) for cat, hs in groups.items() if hs)
