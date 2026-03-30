"""Lazy token label lookup for SAE reference pretraining.

Loads pre-normalized token annotation parquets (24 chromosomes, ~10GB total)
into per-chromosome position→row dicts. At training time, resolves labels
for each SAE token by parsing its genomic position from provenance.

Overhead: ~15s to load all chromosomes, ~15ms per 30K-token chunk lookup.
Per epoch (6210 chunks): ~93s total — negligible vs hours of GPU training.

Usage:
    lookup = TokenLabelLookup(head_names=("phylop_100way", "is_cpg_island", ...))

    # Per chunk: parse provenance, get labels
    labels = lookup.label_chunk(provenance, n_tokens)  # [n_tokens, n_heads] float32
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch

logger = logging.getLogger(__name__)

NORMALIZED_DIR = Path(
    "/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian"
    "/mendelian_saes/bidirectional_saes_v1/sae_datasets/token_annotations_normalized"
)

CHROMOSOMES = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX", "chrY")


class TokenLabelLookup:
    """Position→label lookup across all chromosomes.

    Loads normalized parquets into numpy arrays with position→index dicts
    for O(1) per-token lookup. Uses int8 for binary columns and float16
    for continuous to minimize memory (~60GB for 223M positions × 430 heads).

    Args:
        head_names: Ordered tuple of column names to extract as labels.
        normalized_dir: Path to pre-normalized per-chromosome parquets.
    """

    def __init__(
        self,
        head_names: tuple[str, ...],
        normalized_dir: Path = NORMALIZED_DIR,
    ):
        self.head_names = head_names
        self.n_heads = len(head_names)
        self._pos_to_idx: dict[str, dict[int, int]] = {}
        self._labels: dict[str, np.ndarray] = {}

        logger.info(f"Loading {len(CHROMOSOMES)} normalized token parquets...")
        for chrom in CHROMOSOMES:
            path = normalized_dir / f"{chrom}.parquet"
            if not path.exists():
                continue

            # Only load position + requested columns (fast: parquet column pruning)
            available = set(pl.read_parquet_schema(path).keys())
            load_cols = ["position"] + [h for h in head_names if h in available]
            df = pl.read_parquet(path, columns=load_cols)

            # Build label matrix with compact dtypes
            cols = []
            for name in head_names:
                if name in df.columns:
                    col_data = df[name]
                    if col_data.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                        cols.append(col_data.fill_null(-1).to_numpy().astype(np.int8))
                    else:
                        cols.append(col_data.to_numpy().astype(np.float16))
                else:
                    cols.append(np.full(df.height, np.nan, dtype=np.float16))

            self._labels[chrom] = cols
            self._pos_to_idx[chrom] = dict(zip(df["position"].to_list(), range(df.height)))
            logger.info(f"  {chrom}: {df.height:,} positions")

        total = sum(len(d) for d in self._pos_to_idx.values())
        total_bytes = sum(sum(c.nbytes for c in cols) for cols in self._labels.values())
        logger.info(f"Total: {total:,} positions, {self.n_heads} heads, {total_bytes / 1e9:.0f} GB")

    def label_chunk(self, provenance: dict, n_tokens: int) -> np.ndarray:
        """Resolve labels for one SAE chunk from its provenance.

        Args:
            provenance: Dict with 'sequence_ids' and 'positions' lists.
            n_tokens: Number of tokens in the chunk.

        Returns:
            [n_tokens, n_heads] float32 array. NaN for missing labels,
            -1 for missing binary labels.
        """
        labels = np.full((n_tokens, self.n_heads), np.nan, dtype=np.float32)

        for i in range(n_tokens):
            parts = provenance["sequence_ids"][i].split(":")
            chrom = parts[1]
            genomic_pos = int(parts[2]) + provenance["positions"][i]

            pos_dict = self._pos_to_idx.get(chrom)
            if pos_dict is None:
                continue
            row = pos_dict.get(genomic_pos, -1)
            if row < 0:
                continue

            chrom_cols = self._labels[chrom]
            for j in range(self.n_heads):
                labels[i, j] = float(chrom_cols[j][row])

        return labels


class RefChunkLoader:
    """Stream SAE activations with position-resolved reference labels.

    Wraps goodfire-core's ActivationDataset/training_iterator to yield
    (acts, labels, valid) tuples where labels come from TokenLabelLookup.

    Args:
        sae_dir: Path to SAE activation dataset (goodfire-core chunked format).
        head_names: Ordered tuple of annotation column names to resolve.
        batch_size: Batch size per GPU rank.
        device: Target device string (e.g. "cuda:0").
    """

    def __init__(
        self,
        sae_dir: Path,
        head_names: tuple[str, ...],
        batch_size: int,
        device: str,
    ):
        from goodfire_core.storage import ActivationDataset, FilesystemStorage

        self.device = device
        self.lookup = TokenLabelLookup(head_names=head_names)

        storage = FilesystemStorage(sae_dir.parent)
        self.dataset = ActivationDataset(
            storage,
            sae_dir.name,
            batch_size=batch_size,
            include_provenance=True,
        )

    def iter_epoch(self):
        """Yield (acts, labels, valid) for one epoch over SAE activations.

        acts:   [B, d_model] float on device
        labels: [B, n_heads] float32 on device
        valid:  [B] bool on device — True where at least one label was resolved
        """
        for batch in self.dataset.training_iterator(
            device=self.device, n_epochs=1, shuffle=True,
        ):
            n = batch.acts.shape[0]

            # Build provenance dict for label_chunk
            positions = (
                batch.token_positions.tolist()
                if batch.token_positions is not None
                else [0] * n
            )
            provenance = {
                "sequence_ids": batch.sequence_ids,
                "positions": positions,
            }

            # Resolve labels: [n, n_heads] float32 numpy (NaN = missing)
            labels_np = self.lookup.label_chunk(provenance, n)
            labels = torch.from_numpy(labels_np).to(self.device, non_blocking=True)

            # Valid = at least one non-NaN label per token
            valid = ~torch.isnan(labels).all(dim=1)

            yield batch.acts, labels, valid
