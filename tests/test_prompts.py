"""Tests for Claude interpretation prompt building."""
import sys
from pathlib import Path

# Add the variant-viewer root to sys.path so we can import prompts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts import build_prompt


def _fake_variant(**overrides):
    base = {
        "id": "chr1:100:A:G", "gene": "TEST", "chrom": "chr1", "pos": 100,
        "ref": "A", "alt": "G", "score": 0.75, "label": "VUS",
        "consequence": "missense_variant", "substitution": "A>V",
        "significance": "", "stars": 0, "disease": "",
        "disruption": {"phylop_100way": -0.15, "plddt": 0.05},
        "effect": {"cadd_c": 0.8, "sift_c": 0.02},
        "gt": {"cadd_c": 0.75},
        "neighbors": [{"gene": "BRCA1", "label": "pathogenic", "score": 0.9, "similarity": 0.95}],
        "nP": 1, "nB": 0, "nV": 0,
    }
    base.update(overrides)
    return base


def test_build_prompt_basic():
    prompt = build_prompt(_fake_variant())
    assert "TEST" in prompt
    assert "chr1:100:A:G" in prompt
    assert "75%" in prompt  # score as percentage
    assert "Nearest Neighbors" in prompt


def test_build_prompt_with_attribution():
    v = _fake_variant(attribution={
        "heads": [
            {"name": "phylop_100way", "kind": "disruption", "coefficient": 0.5},
            {"name": "cadd_c", "kind": "effect", "coefficient": 0.3},
        ],
        "explained": 0.8,
    })
    prompt = build_prompt(v)
    assert "Attribution" in prompt
    assert "phylop" in prompt.lower() or "PhyloP" in prompt


def test_build_prompt_without_attribution():
    prompt = build_prompt(_fake_variant(attribution=None))
    assert "None" not in prompt
    assert "Nearest Neighbors" in prompt


def test_hgvs_consistency_drops_mismatched():
    # Consequence=missense but HGVS says synonymous (p.Xxx=)
    v = _fake_variant(hgvsp="ENSP00000123.1:p.Ala100=", hgvsc="ENST00000123.1:c.300A>G")
    prompt = build_prompt(v)
    assert "p.Ala100=" not in prompt  # mismatched HGVS should be dropped


def test_hgvs_consistency_keeps_matching():
    v = _fake_variant(hgvsp="ENSP00000123.1:p.Ala100Val", hgvsc="ENST00000123.1:c.300A>G")
    prompt = build_prompt(v)
    assert "p.Ala100Val" in prompt
