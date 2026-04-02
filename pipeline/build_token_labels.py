"""Build per-head token-level labels as a single safetensors file.

For each variant's 512 positions (256 downstream + 256 upstream), looks up
genomic annotations via vectorized searchsorted. Pure polars + torch, no numpy.

Output:
  token_labels.safetensors  — one key per head, each [N_variants, 512]
                              binary heads: uint8 {0, 1, 255=missing}
                              continuous heads: float16 {value, NaN=missing}
  token_labels.json         — ordered variant IDs + head metadata

Usage:
    uv run python pipeline/build_token_labels.py [ACTIVATIONS_DIR]
"""

import json
import logging
import sys
import time
from pathlib import Path

import polars as pl
import torch
from goodfire_core.storage import FilesystemStorage
from safetensors.torch import save_file
from tqdm import tqdm

from loaders import load_heads
from paths import DECONFOUNDED
from training import local_activation_dataset

logger = logging.getLogger(__name__)

NORMALIZED_DIR = Path(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian"
    "/mendelian_saes/bidirectional_saes_v1/sae_datasets/token_annotations_normalized"
)

MISSING = 255


def _timed(msg: str, t0: float) -> float:
    now = time.time()
    logger.info(f"[{now - t0:.1f}s] {msg}")
    return now


# ── Position loading ─────────────────────────────────────────────────


def load_positions(storage: FilesystemStorage) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Load variant IDs (ordered) and positions as contiguous tensors."""
    ds = local_activation_dataset(storage, "positions", batch_size=512, include_provenance=True)
    ordered_ids: list[str] = []
    fwd_chunks: list[torch.Tensor] = []
    bwd_chunks: list[torch.Tensor] = []
    for chunk_id in tqdm(range(ds.num_chunks), desc="Loading positions"):
        chunk = ds.load_chunk(chunk_id)
        ordered_ids.extend(chunk.sequence_ids)
        fwd_chunks.append(chunk.acts[:, 0])
        bwd_chunks.append(chunk.acts[:, 1])
    return (
        ordered_ids,
        torch.cat(fwd_chunks, dim=0).long(),
        torch.cat(bwd_chunks, dim=0).long(),
    )


# ── Per-chromosome annotation loading ────────────────────────────────


def load_chromosome_annotations(
    path: Path,
    head_names: tuple[str, ...],
    needed_positions: torch.Tensor,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load + filter annotations for one chromosome.

    Args:
        needed_positions: 1D int64 tensor of unique positions to load.
    Returns:
        (sorted_positions [M], label_matrix [M, n_heads])
    """
    available = set(pl.read_parquet_schema(path).keys())
    load_cols = ["position"] + [h for h in head_names if h in available]

    # Semi-join is faster than is_in for large position sets
    needed_df = pl.DataFrame({"position": needed_positions}).lazy()
    df = (
        pl.scan_parquet(path, cache=False)
        .select(load_cols)
        .join(needed_df, on="position", how="semi")
        .sort("position")
        .collect()
    )

    # Add missing columns in one batch (no per-column loop)
    missing = [h for h in head_names if h not in df.columns]
    if missing:
        fill = -1 if kind == "binary" else None
        dtype = pl.Int16 if kind == "binary" else pl.Float32
        df = df.with_columns([pl.lit(fill).cast(dtype).alias(h) for h in missing])

    sorted_positions = df["position"].to_torch().long()

    if kind == "binary":
        head_df = df.select([pl.col(h).fill_null(-1).cast(pl.Int16) for h in head_names])
        raw = head_df.to_torch(dtype=pl.Int16)  # [M, n_heads]
        label_mat = torch.full_like(raw, MISSING, dtype=torch.uint8)
        valid = (raw == 0) | (raw == 1)
        label_mat[valid] = raw[valid].to(torch.uint8)
    else:
        head_df = df.select([pl.col(h).fill_null(float("nan")).cast(pl.Float32) for h in head_names])
        label_mat = head_df.to_torch(dtype=pl.Float32)  # [M, n_heads]

    return sorted_positions, label_mat


# ── Vectorized position lookup ───────────────────────────────────────


def resolve_positions(
    query: torch.Tensor,
    sorted_positions: torch.Tensor,
    label_mat: torch.Tensor,
    n_heads: int,
    kind: str,
) -> torch.Tensor:
    """query [M] → labels [M, n_heads] via searchsorted."""
    dtype = torch.uint8 if kind == "binary" else torch.float32
    fill = MISSING if kind == "binary" else float("nan")
    result = torch.full((len(query), n_heads), fill, dtype=dtype)

    if len(sorted_positions) == 0:
        return result

    indices = torch.searchsorted(sorted_positions, query).clamp(0, len(sorted_positions) - 1)
    matches = sorted_positions[indices] == query
    if matches.any():
        result[matches] = label_mat[indices[matches]]

    return result


# ── Main build ───────────────────────────────────────────────────────


def build_labels(
    ordered_ids: list[str],
    fwd_positions: torch.Tensor,
    bwd_positions: torch.Tensor,
    head_names: tuple[str, ...],
    normalized_dir: Path,
    kind: str,
) -> torch.Tensor:
    """Build [N, 512, n_heads] label tensor. Vectorized per-chromosome."""
    n_variants = len(ordered_ids)
    n_heads = len(head_names)
    dtype = torch.uint8 if kind == "binary" else torch.float32
    fill = MISSING if kind == "binary" else float("nan")

    labels = torch.full((n_variants, 512, n_heads), fill, dtype=dtype)

    # Group variant indices by chromosome
    chrom_to_indices: dict[str, list[int]] = {}
    for i, sid in enumerate(ordered_ids):
        chrom_to_indices.setdefault(sid.split(":")[0], []).append(i)

    for chrom in tqdm(sorted(chrom_to_indices.keys()), desc=f"Building ({kind})"):
        path = normalized_dir / f"{chrom}.parquet"
        if not path.exists():
            logger.warning(f"No annotation file for {chrom}")
            continue

        idx_t = torch.tensor(chrom_to_indices[chrom], dtype=torch.long)

        # Unique positions needed for this chromosome (torch-level, no Python set)
        all_pos = torch.cat([fwd_positions[idx_t].reshape(-1), bwd_positions[idx_t].reshape(-1)])
        unique_pos = all_pos.unique()

        sorted_pos, label_mat = load_chromosome_annotations(path, head_names, unique_pos, kind)

        # Forward: flatten → lookup → unflatten
        fwd_labels = resolve_positions(fwd_positions[idx_t].reshape(-1), sorted_pos, label_mat, n_heads, kind)
        labels[idx_t, :256] = fwd_labels.reshape(len(idx_t), 256, n_heads)

        # Backward
        bwd_labels = resolve_positions(bwd_positions[idx_t].reshape(-1), sorted_pos, label_mat, n_heads, kind)
        labels[idx_t, 256:] = bwd_labels.reshape(len(idx_t), 256, n_heads)

    return labels


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()

    activations = Path(sys.argv[1]) if len(sys.argv) > 1 else DECONFOUNDED
    heads = load_heads()
    storage = FilesystemStorage(activations)

    ordered_ids, fwd_positions, bwd_positions = load_positions(storage)
    _timed(f"Positions: {len(ordered_ids):,} variants", t0)

    binary_names = tuple(
        name for name, info in heads.items() if info["category"] != "effect" and info["kind"] == "binary"
    )
    continuous_names = tuple(
        name for name, info in heads.items() if info["category"] != "effect" and info["kind"] == "continuous"
    )
    logger.info(f"Heads: {len(binary_names)} binary + {len(continuous_names)} continuous")

    binary_labels = build_labels(ordered_ids, fwd_positions, bwd_positions, binary_names, NORMALIZED_DIR, "binary")
    _timed(f"Binary: {binary_labels.shape}", t0)

    continuous_labels = None
    if continuous_names:
        continuous_labels = build_labels(
            ordered_ids, fwd_positions, bwd_positions, continuous_names, NORMALIZED_DIR, "continuous"
        )
        _timed(f"Continuous: {continuous_labels.shape}", t0)

    # ── Validate and filter: drop heads with degenerate data ────────
    kept_binary: list[str] = []
    kept_continuous: list[str] = []
    dropped: list[str] = []
    all_tensors: dict[str, torch.Tensor] = {}

    for h_idx, name in enumerate(binary_names):
        t = binary_labels[:, :, h_idx].contiguous()
        valid = t != MISSING
        coverage = valid.sum().item() / t.numel()
        if coverage == 0:
            dropped.append(f"{name} (binary: no valid data at all)")
            continue
        all_tensors[name] = t
        kept_binary.append(name)

    if continuous_labels is not None:
        for h_idx, name in enumerate(continuous_names):
            t = continuous_labels[:, :, h_idx]
            valid = ~torch.isnan(t)
            coverage = valid.sum().item() / t.numel()
            std = t[valid].std().item() if valid.any() else 0.0
            if coverage < 0.01 or std < 1e-6:
                dropped.append(f"{name} (continuous: coverage={coverage:.1%}, std={std:.4f})")
                continue
            all_tensors[name] = t.to(torch.float16).contiguous()
            kept_continuous.append(name)

    if dropped:
        logger.warning(f"Dropped {len(dropped)} degenerate heads:")
        for d in dropped:
            logger.warning(f"  {d}")

    logger.info(f"Kept: {len(kept_binary)} binary + {len(kept_continuous)} continuous = {len(all_tensors)}")
    _timed("Validated and sliced", t0)

    # ── Save ─────────────────────────────────────────────────────────
    assert len(all_tensors) > 0, "No heads passed validation"

    out_path = activations / "token_labels.safetensors"
    save_file(all_tensors, str(out_path))

    meta = {
        "ids": ordered_ids,
        "binary_heads": kept_binary,
        "continuous_heads": kept_continuous,
    }
    (activations / "token_labels.json").write_text(json.dumps(meta))

    total_gb = sum(t.nbytes for t in all_tensors.values()) / 1e9
    _timed(f"Done: {out_path} ({total_gb:.1f} GB, {len(all_tensors)} heads)", t0)


if __name__ == "__main__":
    main()
