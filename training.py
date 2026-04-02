"""Training utilities shared by sequence-level and token-level probe training."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import polars as pl
import torch
import torch.distributed as dist
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from goodfire_core.training.callbacks import CheckpointCallback
from loguru import logger

from loaders import load_heads
from probe.covariance import HeadSpec


# ── NFS workaround ─────────────────────────────────────────────────────


def local_activation_dataset(storage: FilesystemStorage, name: str, **kwargs) -> ActivationDataset:
    """Create an ActivationDataset with SQLite index copied to local /tmp.

    SQLite WAL-mode locking deadlocks on NFS when multiple parallel jobs
    (SLURM array shards or DDP ranks) access the same index file. Copying
    to /tmp gives each process its own read-only copy.
    """
    ds = ActivationDataset(storage, name, **kwargs)
    # Attribute name varies across goodfire-core versions
    attr = "_index_sqlite_path" if hasattr(ds, "_index_sqlite_path") else "_index_path"
    index_path = getattr(ds, attr, None)
    if index_path:
        local = Path(tempfile.mkdtemp(prefix="gf_idx_")) / "index.sqlite"
        shutil.copy2(index_path, local)
        setattr(ds, attr, str(local))
    return ds


# ── Training callbacks (goodfire-core integration) ──────────────────────


def setup_training_callbacks(
    project: str,
    name: str,
    config: dict,
    checkpoint_dir: str,
    rank: int = 0,
) -> list:
    """Create goodfire-core WandBCallback + CheckpointCallback for rank 0.

    Returns an empty list for non-zero ranks (no logging/checkpointing).
    WandB failure is non-fatal — training continues without experiment tracking.
    """
    if rank != 0:
        return []

    callbacks: list = [
        CheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            save_every_epochs=1,
            save_best=False,
        ),
    ]

    try:
        from goodfire_core.training.callbacks import WandBCallback, WandBConfig

        callbacks.append(WandBCallback(WandBConfig(project=project, name=name), config_dict=config))
    except Exception as exc:
        logger.warning(f"WandB callback init failed (training continues): {exc}")

    return callbacks


def fire_callbacks(callbacks: list, method: str, **kwargs) -> None:
    """Call a callback method on all callbacks."""
    for cb in callbacks:
        fn = getattr(cb, method, None)
        if fn is not None:
            fn(**kwargs)


# ── Head loading ─────────────────────────────────────────────────────────

# Columns that are positional metadata, not prediction targets.
# These get auto-discovered by discover_heads() as categorical heads
# but are indices/counts, not biological properties worth predicting.
_METADATA_HEADS = frozenset(
    {
        "residue_number",  # amino acid index in protein (up to 35990)
        "exon_number",  # exon index in transcript (up to 364)
        "n_transcripts_with_exon",  # transcript count (up to 350)
        "ppi_partner_count",  # interaction partner count (up to 502)
        "fstack_state",  # chromatin state enum (101 states)
    }
)


def load_head_specs() -> tuple[dict[str, HeadSpec], dict[str, HeadSpec]]:
    """Load heads split into (disruption, effect) from heads.json.

    Excludes metadata heads (positional indices, counts) that were
    auto-discovered as categorical but aren't meaningful prediction targets.
    """
    heads = load_heads()
    disruption, effect = {}, {}
    for name, info in heads.items():
        if name in _METADATA_HEADS:
            continue
        spec = HeadSpec(n_classes=info["n_classes"], kind=info["kind"])
        if info["category"] == "effect":
            effect[name] = spec
        else:
            disruption[name] = spec
    return disruption, effect


def load_disruption_specs() -> dict[str, HeadSpec]:
    """Load only disruption heads from heads.json."""
    disruption, _ = load_head_specs()
    return disruption


# ── Gene split ───────────────────────────────────────────────────────────


def gene_split(
    metadata: pl.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    gene_col: str = "gene_name",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split metadata by genes — no gene appears in both sets.

    Gene-level splitting prevents data leakage from shared gene effects.
    Deterministic: same (metadata, test_size, seed) → same split.
    """
    genes = metadata.select(gene_col).unique().sort(gene_col)
    n_test = int(len(genes) * test_size)
    shuffled = genes.sample(fraction=1.0, seed=seed, shuffle=True)
    test_genes = set(shuffled.head(n_test)[gene_col].to_list())

    test_mask = metadata[gene_col].is_in(list(test_genes))
    return metadata.filter(~test_mask), metadata.filter(test_mask)


# ── DDP context ──────────────────────────────────────────────────────────


@contextmanager
def ddp_context(timeout_minutes: int = 30) -> Generator[tuple[torch.device, int, int], None, None]:
    """Initialize DDP if RANK env var is set, yield (device, rank, world_size), clean up.

    Usage:
        with ddp_context() as (device, rank, world_size):
            probe = probe.to(device)
            if world_size > 1:
                probe = DistributedDataParallel(probe, device_ids=[rank])
            # ... training loop
    """
    distributed = dist.is_available() and "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(minutes=timeout_minutes))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda")
    try:
        yield device, rank, world_size
    finally:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
