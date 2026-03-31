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
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import typer
from goodfire_core.data.interfaces import TensorActivations
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from goodfire_core.training.optimizers import EMuon
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import wandb
from paths import ARTIFACTS, DATA
from pipeline.extract import unified_diff, unified_ref
from probe.covariance import MultiHeadCovProbeV2, multihead_loss_v2
from training import gene_split, load_head_specs

# ── Paths ────────────────────────────────────────────────────────────────

DECONFOUNDED = ARTIFACTS / "clinvar_evo2_deconfounded_full"


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

    # DDP
    distributed = dist.is_available() and "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda")

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

    label_cols = []
    for h in head_names:
        if h in labeled.columns:
            label_cols.append(labeled[h].to_numpy().astype("float32"))
        else:
            spec = all_specs[h]
            fill = float("nan") if spec.kind == "continuous" else -1.0
            label_cols.append(np.full(labeled.height, fill, dtype=np.float32))
    label_matrix = torch.from_numpy(np.stack(label_cols, axis=1))
    id_to_idx = {vid: i for i, vid in enumerate(labeled["variant_id"].to_list())}

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
    dataset = ActivationDataset(storage, "activations", batch_size=per_gpu)
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
        d_model=d_model, heads=all_specs, d_hidden=d_hidden, d_probe=d_probe,
    ).to(device)
    optimizer = EMuon(probe.parameters(), lr=lr)

    if distributed:
        probe = DistributedDataParallel(probe, device_ids=[rank])

    if rank == 0:
        wandb.init(project="gfm-probes", name=name, config={
            "preset": preset, "seed": seed, "test_size": test_size,
            "d_model": d_model, "d_hidden": d_hidden, "d_probe": d_probe,
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "focal_gamma": focal_gamma,
            "n_disruption_heads": len(disruption_specs),
            "n_effect_heads": len(effect_specs),
        })
        logger.info(f"Train: {len(train_ids):,} variants, {iterator.steps_per_epoch} steps/epoch")

    # Train
    for epoch in range(epochs):
        probe.train()
        pbar = tqdm(iterator.iter_epoch(), total=iterator.steps_per_epoch,
                    desc=f"Epoch {epoch+1}/{epochs}", disable=rank != 0)

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
                probe(unified_diff(raw)), diff_labels, specs_tuple,
                focal_gamma=focal_gamma,
            )

            # Pass 2: ref → disruption heads
            loss_ref = multihead_loss_v2(
                probe(unified_ref(raw)), ref_labels, specs_tuple,
                focal_gamma=focal_gamma,
            )

            loss = loss_diff + loss_ref
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}", diff=f"{loss_diff.item():.3f}", ref=f"{loss_ref.item():.3f}")
                wandb.log({"loss": loss.item(), "loss_diff": loss_diff.item(), "loss_ref": loss_ref.item()})

        if rank == 0:
            raw_p = probe.module if distributed else probe
            raw_p.save_checkpoint(str(out_dir / f"checkpoint_epoch_{epoch+1}.pt"))

    # Save
    if rank == 0:
        raw_p = probe.module if distributed else probe
        raw_p.save_checkpoint(str(out_dir / "weights.pt"))
        (out_dir / "config.json").write_text(json.dumps({
            "preset": preset, "seed": seed, "test_size": test_size,
            "d_model": d_model, "d_hidden": d_hidden, "d_probe": d_probe,
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "training": "dual_pass",
            "effect_heads_view": "diff",
            "disruption_heads_view": "ref",
            "disruption_heads": list(disruption_specs.keys()),
            "effect_heads": list(effect_specs.keys()),
            "n_disruption_heads": len(disruption_specs),
            "n_effect_heads": len(effect_specs),
            "n_train": train_df.height, "n_test": test_df.height,
            "focal_gamma": focal_gamma,
        }, indent=2))
        wandb.finish()
        logger.info(f"Saved: {out_dir / 'weights.pt'}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


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
