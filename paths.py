"""Path constants and ID utilities for the variant viewer.

Centralizes:
- Artifact path constants
- Variant ID sanitization (must match between build.py, serve.py, and JS frontend)
"""

from pathlib import Path

# ── Artifact roots ────────────────────────────────────────────────────────
ARTIFACTS = Path("/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian")
MAYO_DATA = Path(__file__).parent / "data"
VEP_DOMAIN_CACHE = ARTIFACTS.parent / "annotations" / "sources" / "cache" / "vep_domain_lookup.json"


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
