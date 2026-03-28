"""Smoke tests for the variant viewer build pipeline."""
import json
from pathlib import Path

import pytest

# Use the latest staging dir if it exists, otherwise skip
STAGING_DIRS = sorted(Path("/tmp").glob("variant_viewer_*"), key=lambda p: p.stat().st_mtime, reverse=True)
BUILD_DIR = STAGING_DIRS[0] if STAGING_DIRS else None


@pytest.fixture
def variant():
    if not BUILD_DIR or not (BUILD_DIR / "variants").exists():
        pytest.skip("No build found in /tmp")
    f = next((BUILD_DIR / "variants").glob("*.json"))
    return json.loads(f.read_bytes())


@pytest.fixture
def global_data():
    if not BUILD_DIR or not (BUILD_DIR / "global.json").exists():
        pytest.skip("No build found in /tmp")
    return json.loads((BUILD_DIR / "global.json").read_bytes())


def test_variant_required_keys(variant):
    required = {"id", "gene", "chrom", "pos", "ref", "alt", "score", "label",
                "consequence", "disruption", "effect", "neighbors"}
    assert required <= set(variant.keys()), f"Missing: {required - set(variant.keys())}"


def test_variant_score_range(variant):
    assert 0 <= variant["score"] <= 1, f"Score out of range: {variant['score']}"


def test_variant_disruption_is_scalar(variant):
    for head, val in variant.get("disruption", {}).items():
        assert isinstance(val, (int, float)), f"Disruption {head} should be scalar, got {type(val)}"


def test_global_distributions(global_data):
    assert "distributions" in global_data
    assert "pathogenic" in global_data["distributions"]
    dist = global_data["distributions"]["pathogenic"]
    assert "benign" in dist and "pathogenic" in dist and "bins" in dist


def test_global_heads(global_data):
    assert "heads" in global_data
    assert "disruption" in global_data["heads"]
    assert "effect" in global_data["heads"]


def test_global_display_names(global_data):
    display = global_data.get("display", {})
    heads = global_data.get("heads", {})
    all_heads = set()
    for group_heads in heads.get("disruption", {}).values():
        all_heads.update(group_heads)
    for group_heads in heads.get("effect", {}).values():
        all_heads.update(group_heads)
    for h in all_heads:
        assert h in display, f"Head {h} missing display name"
