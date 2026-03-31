"""Path constants and ID utilities for the variant viewer."""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────

ARTIFACTS = Path(os.environ.get(
    "VV_ARTIFACTS",
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian",
))
DATA = Path(__file__).parent / "data"
VARIANTS = DATA / "variants.parquet"
HEADS = DATA / "heads.json"


# ── ID utilities ─────────────────────────────────────────────────────────

def sanitize_vid(v: str) -> str:
    """Make a variant ID safe for use as a filename.

    Uses FNV-1a so the JS frontend can compute the same hash synchronously.
    """
    s = v.replace(":", "_").replace("/", "_")
    if len(s) <= 200:
        return s
    h = 0xCBF29CE484222325
    for b in v.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{s[:60]}_{h:016x}"
