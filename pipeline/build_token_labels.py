"""Build binary token-level labels aligned to the variant activation dataset.

For each variant's 512 positions (256 downstream + 256 upstream), looks up
binary genomic annotations via vectorized searchsorted. Stores as uint8
{0=negative, 1=positive, 255=missing}.

Only binary disruption heads. Continuous and categorical heads are excluded.

Usage:
    uv run python pipeline/build_token_labels.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from tqdm import tqdm

from loaders import load_heads
from paths import DECONFOUNDED

logger = logging.getLogger(__name__)

NORMALIZED_DIR = Path(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian"
    "/mendelian_saes/bidirectional_saes_v1/sae_datasets/token_annotations_normalized"
)

MISSING: np.uint8 = np.uint8(255)


def load_positions(storage: FilesystemStorage) -> dict[str, torch.Tensor]:
    """Load all positions: sequence_id → [2, 256] int64."""
    pos_ds = ActivationDataset(storage, "positions", batch_size=512, include_provenance=True)
    pos_dict: dict[str, torch.Tensor] = {}
    for chunk_id in tqdm(range(pos_ds.num_chunks), desc="Loading positions"):
        chunk = pos_ds.load_chunk(chunk_id)
        for i, sid in enumerate(chunk.sequence_ids):
            pos_dict[sid] = chunk.acts[i]
    logger.info(f"Positions: {len(pos_dict):,} variants")
    return pos_dict


def build_label_index(
    pos_dict: dict[str, torch.Tensor],
    head_names: tuple[str, ...],
    normalized_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build per-chromosome (sorted_positions, label_matrix) for vectorized lookup.

    Returns {chrom: (sorted_positions[N], labels[N, n_heads] uint8)}.
    Only positions needed by our variants are loaded (predicate pushdown).
    """
    needed: dict[str, set[int]] = {}
    for sid, positions in pos_dict.items():
        chrom = sid.split(":")[0]
        s = needed.setdefault(chrom, set())
        s.update(positions[0].tolist())
        s.update(positions[1].tolist())

    total_needed = sum(len(v) for v in needed.values())
    logger.info(f"Need {total_needed:,} unique positions across {len(needed)} chromosomes")

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for chrom in tqdm(sorted(needed.keys()), desc="Loading annotations"):
        path = normalized_dir / f"{chrom}.parquet"
        if not path.exists():
            continue

        positions_set = needed[chrom]
        available = set(pl.read_parquet_schema(path).keys())
        load_cols = ["position"] + [h for h in head_names if h in available]

        df = (
            pl.scan_parquet(path, cache=False)
            .select(load_cols)
            .filter(pl.col("position").is_in(list(positions_set)))
            .sort("position")
            .collect()
        )

        n_rows = df.height
        n_heads = len(head_names)
        label_mat = np.full((n_rows, n_heads), MISSING, dtype=np.uint8)

        for h_idx, h in enumerate(head_names):
            if h not in df.columns:
                continue
            arr = df[h].fill_null(-1).to_numpy()
            valid = (arr == 0) | (arr == 1)
            out = np.full(len(arr), MISSING, dtype=np.uint8)
            out[valid] = arr[valid].astype(np.uint8)
            label_mat[:, h_idx] = out

        sorted_positions = df["position"].to_numpy().astype(np.int64)
        result[chrom] = (sorted_positions, label_mat)

        n_valid = np.sum(label_mat != MISSING)
        logger.info(f"  {chrom}: {n_rows:,} positions, {n_valid / label_mat.size:.1%} coverage")

    mem_gb = sum(arr.nbytes + mat.nbytes for arr, mat in result.values()) / 1e9
    logger.info(f"Label index: {mem_gb:.1f} GB")
    return result


def resolve_labels(
    positions: np.ndarray,
    chrom_entry: tuple[np.ndarray, np.ndarray],
    n_heads: int,
) -> np.ndarray:
    """Resolve labels for a 1D array of genomic positions → [n_heads, len(positions)] uint8."""
    result = np.full((n_heads, len(positions)), MISSING, dtype=np.uint8)
    sorted_positions, label_mat = chrom_entry

    indices = np.searchsorted(sorted_positions, positions)
    indices = np.clip(indices, 0, len(sorted_positions) - 1)
    matches = sorted_positions[indices] == positions

    matched_indices = indices[matches]
    if matched_indices.size > 0:
        result[:, matches] = label_mat[matched_indices].T

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    activations = Path(sys.argv[1]) if len(sys.argv) > 1 else DECONFOUNDED
    output_name = sys.argv[2] if len(sys.argv) > 2 else "token_labels_binary"

    heads = load_heads()
    head_names = tuple(
        name for name, info in heads.items()
        if info["category"] != "effect" and info["kind"] == "binary"
    )
    n_heads = len(head_names)
    logger.info(f"Binary disruption heads: {n_heads}")

    storage = FilesystemStorage(activations)
    pos_dict = load_positions(storage)
    chrom_data = build_label_index(pos_dict, head_names, NORMALIZED_DIR)

    # Import build_aligned_dataset from annotator
    annotator_scripts = Path(__file__).resolve().parent.parent.parent / "annotator" / "scripts"
    sys.path.insert(0, str(annotator_scripts))
    from build_aligned_dataset import build_aligned_dataset

    def transform(sequence_id: str, ref_item: torch.Tensor) -> torch.Tensor:
        positions = pos_dict.get(sequence_id)
        if positions is None:
            return torch.full((n_heads, 512), MISSING, dtype=torch.uint8)

        chrom = sequence_id.split(":")[0]
        entry = chrom_data.get(chrom)
        if entry is None:
            return torch.full((n_heads, 512), MISSING, dtype=torch.uint8)

        fwd_pos = positions[0].numpy().astype(np.int64)
        bwd_pos = positions[1].numpy().astype(np.int64)

        fwd_labels = resolve_labels(fwd_pos, entry, n_heads)
        bwd_labels = resolve_labels(bwd_pos, entry, n_heads)

        return torch.from_numpy(np.concatenate([fwd_labels, bwd_labels], axis=1))

    out_dir = build_aligned_dataset(
        storage=storage,
        reference="activations",
        output=output_name,
        transform=transform,
        item_shape=(n_heads, 512),
        dtype="uint8",
        overwrite=True,
    )

    (out_dir / "head_names.json").write_text(json.dumps(list(head_names)))
    logger.info(f"Done: {n_heads} heads × 512 positions, saved to {out_dir}")


if __name__ == "__main__":
    main()
