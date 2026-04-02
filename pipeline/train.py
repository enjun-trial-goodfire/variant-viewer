"""Train dual-pass multihead covariance probe.

Simple two-pass training:
  1. effect heads on DIFF activations (variant-effect signal)
  2. disruption heads on REF activations (positional signal)

Reads head classification from heads.json (effect vs disruption).
Uses focal loss (gamma=0.5) for binary heads.

Usage:
    sbatch --gpus=4 pipeline/train.sh --name probe_v11
"""

import json
import os
from pathlib import Path

import polars as pl
import torch
import torch.distributed as dist
import typer
from goodfire_core.data.interfaces import TensorActivations
from goodfire_core.storage import FilesystemStorage
from goodfire_core.training.optimizers import EMuon
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from paths import DATA, DECONFOUNDED
from pipeline.extract import unified_diff, unified_ref
from probe.covariance import MultiHeadCovProbeV2, multihead_loss_v2
from training import (
    ddp_context,
    fire_callbacks,
    gene_split,
    load_head_specs,
    local_activation_dataset,
    setup_training_callbacks,
)


# ── Training ─────────────────────────────────────────────────────────────


def train(
    name: str,
    activations: Path,
    preset: str,
    d_model: int,
    d_hidden: int,
    d_probe: int,
    epochs: int,
    lr: float,
    batch_size: int,
    test_size: float,
    seed: int,
    focal_gamma: float = 0.5,
) -> None:
    torch.manual_seed(seed)

    with ddp_context() as (device, rank, world_size):
        distributed = world_size > 1
        _train_inner(
            name,
            activations,
            preset,
            d_model,
            d_hidden,
            d_probe,
            epochs,
            lr,
            batch_size,
            test_size,
            seed,
            focal_gamma,
            device,
            rank,
            world_size,
            distributed,
        )


def _train_inner(
    name,
    activations,
    preset,
    d_model,
    d_hidden,
    d_probe,
    epochs,
    lr,
    batch_size,
    test_size,
    seed,
    focal_gamma,
    device,
    rank,
    world_size,
    distributed,
):
    # Heads
    disruption_specs, effect_specs = load_head_specs()
    all_specs = {**disruption_specs, **effect_specs}
    head_names = list(all_specs.keys())
    disruption_indices = [i for i, n in enumerate(head_names) if n in disruption_specs]
    effect_indices = [i for i, n in enumerate(head_names) if n in effect_specs]
    specs_tuple = tuple(all_specs.values())

    if rank == 0:
        logger.info(f"Heads: {len(disruption_specs)} disruption + {len(effect_specs)} effect = {len(all_specs)}")

    # Labels — read from preset parquet (e.g., deconfounded-full.parquet)
    labeled = pl.read_parquet(DATA / f"{preset}.parquet")
    train_df, test_df = gene_split(labeled, test_size=test_size, seed=seed)
    train_ids = train_df["variant_id"].to_list()

    # Build label matrix: bulk polars → torch (no per-column .to_list())
    present = [h for h in head_names if h in labeled.columns]
    missing_heads = [h for h in head_names if h not in labeled.columns]
    fill_exprs = []
    for h in missing_heads:
        spec = all_specs[h]
        fill = float("nan") if spec.kind == "continuous" else -1.0
        fill_exprs.append(pl.lit(fill).cast(pl.Float32).alias(h))
    label_df = labeled.select(
        [pl.col(h).fill_null(float("nan")).cast(pl.Float32) for h in present] + fill_exprs
    ).select(list(head_names))  # reorder to match head_names
    label_matrix = label_df.to_torch(dtype=pl.Float32)  # [N, n_heads]
    id_to_idx = dict(zip(labeled["variant_id"].to_physical().to_list(), range(labeled.height)))

    missing = torch.full((len(head_names),), float("nan"), dtype=torch.float32)
    for i, spec in enumerate(specs_tuple):
        if spec.kind != "continuous":
            missing[i] = -1

    # Output
    out_dir = activations / name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            [{"variant_id": v, "split": "train"} for v in train_ids]
            + [{"variant_id": v, "split": "test"} for v in test_df["variant_id"].to_list()]
        ).write_ipc(out_dir / "split.feather")

    if distributed:
        dist.barrier()

    # Iterator
    storage = FilesystemStorage(activations)
    per_gpu = max(1, batch_size // world_size)
    dataset = local_activation_dataset(storage, "activations", batch_size=per_gpu)
    iterator = dataset.training_iterator(device=str(device), n_epochs=epochs, sequence_ids=train_ids)

    # Sync step counts across DDP ranks to prevent deadlocks from uneven chunk sizes.
    # Workaround for goodfire-ai/goodfire-core#293 (not yet merged).
    if distributed:
        steps = torch.tensor([iterator.steps_per_epoch], device=device)
        dist.all_reduce(steps, op=dist.ReduceOp.MIN)
        iterator.set_max_steps(int(steps.item()))

    def inject_labels(batch):
        raw = batch.acts.float()
        indices = [id_to_idx.get(sid, -1) for sid in batch.sequence_ids]
        idx_t = torch.tensor(indices, dtype=torch.long)
        known = idx_t >= 0
        labels = missing.unsqueeze(0).expand(len(indices), -1).clone()
        if known.any():
            labels[known] = label_matrix[idx_t[known]]
        labels = labels.to(raw.device, non_blocking=True)
        return TensorActivations(acts=raw, labels=labels, sequence_ids=batch.sequence_ids)

    iterator.add_transform(inject_labels)

    # Probe
    probe = MultiHeadCovProbeV2(
        d_model=d_model,
        heads=all_specs,
        d_hidden=d_hidden,
        d_probe=d_probe,
    ).to(device)
    optimizer = EMuon(probe.parameters(), lr=lr)

    if distributed:
        probe = DistributedDataParallel(probe, device_ids=[rank])

    config = {
        "preset": preset,
        "seed": seed,
        "test_size": test_size,
        "d_model": d_model,
        "d_hidden": d_hidden,
        "d_probe": d_probe,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "focal_gamma": focal_gamma,
        "training": "dual_pass",
        "n_disruption_heads": len(disruption_specs),
        "n_effect_heads": len(effect_specs),
    }
    callbacks = setup_training_callbacks(
        project="gfm-probes",
        name=name,
        config=config,
        checkpoint_dir=str(out_dir),
        rank=rank,
    )
    fire_callbacks(callbacks, "on_train_begin")

    if rank == 0:
        logger.info(f"Train: {len(train_ids):,} variants, {iterator.steps_per_epoch} steps/epoch")

    # Train
    raw_p = probe.module if distributed else probe  # unwrap once, reuse everywhere
    step = 0
    for epoch in range(epochs):
        probe.train()
        fire_callbacks(callbacks, "on_epoch_begin", epoch=epoch)
        pbar = tqdm(
            iterator.iter_epoch(), total=iterator.steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}", disable=rank != 0
        )
        epoch_loss = 0.0
        epoch_samples = 0

        for batch in pbar:
            if batch.batch_size == 0 or batch.labels is None:
                continue

            raw = batch.acts
            labels = batch.labels

            # Mask labels by view
            ref_labels = labels.clone()
            diff_labels = labels.clone()
            for i in effect_indices:
                ref_labels[:, i] = missing[i]
            for i in disruption_indices:
                diff_labels[:, i] = missing[i]

            # Pass 1: diff → effect heads
            loss_diff = multihead_loss_v2(
                probe(unified_diff(raw)),
                diff_labels,
                specs_tuple,
                focal_gamma=focal_gamma,
            )

            # Pass 2: ref → disruption heads
            loss_ref = multihead_loss_v2(
                probe(unified_ref(raw)),
                ref_labels,
                specs_tuple,
                focal_gamma=focal_gamma,
            )

            loss = loss_diff + loss_ref
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            epoch_loss += loss.item() * batch.batch_size
            epoch_samples += batch.batch_size

            if rank == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}", diff=f"{loss_diff.item():.3f}", ref=f"{loss_ref.item():.3f}"
                )
                fire_callbacks(
                    callbacks, "on_step_end", step=step, loss=loss.item(), batch_size=batch.batch_size, log_frequency=1
                )

        fire_callbacks(
            callbacks, "on_epoch_end", model=raw_p, epoch=epoch, train_loss=epoch_loss / max(epoch_samples, 1)
        )

    # Save final weights + config
    if rank == 0:
        raw_p.save_checkpoint(str(out_dir / "weights.pt"))
        config.update(
            {
                "effect_heads_view": "diff",
                "disruption_heads_view": "ref",
                "disruption_heads": list(disruption_specs.keys()),
                "effect_heads": list(effect_specs.keys()),
                "n_train": train_df.height,
                "n_test": test_df.height,
            }
        )
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        logger.info(f"Saved: {out_dir / 'weights.pt'}")

    fire_callbacks(callbacks, "on_train_end", model=raw_p, epoch=epochs - 1)


# ── CLI ──────────────────────────────────────────────────────────────────

app = typer.Typer(help="Train multihead covariance probe (dual-pass)")


@app.command()
def main(
    name: str = typer.Option("probe_v12", help="Output directory name + wandb run name"),
    activations: Path = typer.Option(DECONFOUNDED, help="Activations storage directory"),
    preset: str = typer.Option("deconfounded-full"),
    d_model: int = typer.Option(8192),
    d_hidden: int = typer.Option(64),
    d_probe: int = typer.Option(128),
    epochs: int = typer.Option(1),
    lr: float = typer.Option(0.01),
    batch_size: int = typer.Option(256),
    test_size: float = typer.Option(0.2),
    seed: int = typer.Option(42),
    focal_gamma: float = typer.Option(0.5, help="Focal loss gamma (0=standard CE, >0=focal loss)"),
) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        os.environ.setdefault("WANDB_DIR", f"/tmp/wandb_{os.environ['SLURM_JOB_ID']}")
    train(**{k: v for k, v in locals().items()})


if __name__ == "__main__":
    app()
