"""Display constants for the variant viewer.

Maps probe head keys to human-readable names and category groups.
No external dependencies beyond the standard library.
"""

# ── Display name overrides ────────────────────────────────────────────────
_DISPLAY_OVERRIDES: dict[str, str] = {
    "cadd_c": "CADD", "revel_c": "REVEL", "alphamissense_c": "AlphaMissense",
    "sift_c": "SIFT", "polyphen_c": "PolyPhen-2", "eve_c": "EVE",
    "spliceai_max_c": "SpliceAI", "gnomad_af_c": "gnomAD AF",
    "spliceai_ag_c": "SpliceAI AG", "spliceai_al_c": "SpliceAI AL",
    "spliceai_dg_c": "SpliceAI DG", "spliceai_dl_c": "SpliceAI DL",
    "mpc_c": "MPC", "mvp_c": "MVP", "mcap_c": "M-CAP",
    "metalr_c": "MetaLR", "vest4_c": "VEST4", "primateai_c": "PrimateAI",
    "mutpred_c": "MutPred", "clinpred_c": "ClinPred", "deogen2_c": "DEOGEN2",
    "bayesdel_c": "BayesDel", "remm_c": "ReMM", "regulomedb_c": "RegulomeDB",
    "phylop_c": "PhyloP", "phastcons_c": "PhastCons", "gerp_c": "GERP",
    "blosum62_c": "BLOSUM62", "grantham_c": "Grantham",
    "hydrophobicity_c": "Hydrophobicity", "volume_c": "Volume", "mw_c": "Mol. Weight",
    "loeuf_c": "LOEUF",
    "gtex_max_tpm_c": "GTEx Max TPM", "gtex_n_tissues_c": "GTEx Tissues",
    "gtex_brain_max_c": "GTEx Brain Max",
    "in_domain": "In Domain", "is_exonic": "Exonic",
    "splice_disrupting": "Splice Disrupting", "charge_altering": "Charge Altering",
    "pathogenic": "Pathogenicity", "consequence": "Consequence", "impact": "Impact",
    "aa_swap": "AA Substitution",
    # AlphaFold (avoid confusion with splicing)
    "psi": "AlphaFold Psi Angle", "phi": "AlphaFold Phi Angle",
    "plddt": "AlphaFold Confidence (pLDDT)", "rsa": "Relative Solvent Accessibility",
    "secondary_structure_H": "Alpha Helix", "secondary_structure_E": "Beta Strand",
    "secondary_structure_C": "Coil/Loop",
    "is_disordered": "Intrinsically Disordered", "is_buried": "Buried Residue",
    # Splicing
    "is_splice_donor": "Splice Donor Site", "is_splice_acceptor": "Splice Acceptor Site",
    "is_branchpoint_region": "Branchpoint Region", "is_polypyrimidine_tract": "Polypyrimidine Tract",
    "is_exon_to_intron": "Exon-Intron Boundary", "is_intron_to_exon": "Intron-Exon Boundary",
    # Genomic context
    "phylop_100way": "PhyloP 100-way", "phastcons_100way": "PhastCons 100-way",
    "gc_content": "GC Content", "cpg_density": "CpG Density",
    "recomb_rate": "Recombination Rate", "codon_position": "Codon Position",
    "exon_number": "Exon Number", "cadd_wg_c": "CADD (whole-genome)",
    "is_ppi_interface": "PPI Interface", "ppi_partner_count": "PPI Partners",
}

_ACRONYMS = {"chipseq", "atacseq", "chromhmm", "fstack", "ptm", "elm"}
_GROUP_PREFIXES = (
    "interpro_", "pfam_", "amino_acid_", "elm_", "gtex_",
    "chipseq_", "atacseq_", "chromhmm_", "fstack_", "ptm_",
)

# ── Head grouping (prefix -> category) ────────────────────────────────────
_PREFIX_TO_GROUP: dict[str, str] = {}
for _group, _prefixes in {
    "Conservation": ("phylop", "phastcons", "gerp"),
    "Protein features": ("secondary", "disorder", "plddt", "rsa", "sasa", "phi", "psi", "ppi", "residue"),
    "Structure": ("in", "is", "has"),
    "InterPro domains": ("interpro",),
    "Pfam domains": ("pfam",),
    "ELM motifs": ("elm",),
    "ChIP-seq": ("chipseq",),
    "ATAC-seq": ("atacseq",),
    "Chromatin": ("chromhmm", "fstack"),
    "Regulatory": ("ccre", "dna"),
    "Amino acid": ("amino",),
    "Sequence context": ("codon", "trinuc", "gc", "cpg", "syn"),
    "Substitution": ("aa", "blosum62", "grantham", "hydrophobicity", "volume", "mw"),
    "Modifications": ("ptm",),
    "Expression": ("gtex",),
    "Region": ("region", "exon", "n", "trf", "segdup", "recomb"),
    "Constraint": ("loeuf",),
    "Clinical": (
        "cadd", "revel", "alphamissense", "sift", "polyphen", "eve", "bayesdel",
        "clinpred", "deogen2", "mcap", "metalr", "mpc", "mutpred", "mvp",
        "primateai", "vest4", "remm", "regulomedb",
    ),
    "Splice": ("spliceai",),
    "Variant effects": ("splice", "charge", "consequence", "impact", "csq"),
    "Population": ("gnomad",),
    "Pathogenicity": ("pathogenic",),
}.items():
    _PREFIX_TO_GROUP.update({p: _group for p in _prefixes})


def display_name(h: str) -> str:
    """Human-readable head name, with group prefix stripped."""
    if h in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[h]
    base = h.removesuffix("_c")
    for prefix in _GROUP_PREFIXES:
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    base = base.replace("_", " ")
    if h.split("_")[0] in _ACRONYMS:
        return base.upper()
    if base.startswith("PF"):
        return base
    return base.title()


def auto_group(heads: set[str]) -> dict[str, list[str]]:
    """Group heads by first-token prefix lookup."""
    g: dict[str, list[str]] = {}
    for h in sorted(heads):
        g.setdefault(_PREFIX_TO_GROUP.get(h.split("_")[0], "Other"), []).append(h)
    return g
