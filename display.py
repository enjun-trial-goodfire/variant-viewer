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
