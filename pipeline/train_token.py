"""Train per-token reference probe by unpacking sequence activations on the fly.

Streams the existing [2, 3, 256, 4096] activations, unpacks to per-token
vectors, resolves annotations from token_annotations_normalized, trains.

Only disruption (reference) heads. No sqrtm (bilinear probe). Summing
token logits at inference = sequence-level outer product (linearity).

Dataset size: 184K variants × 256 tokens = 47M samples per epoch.
Memory: positions dict ~750MB RAM, label lookup ~60GB RAM.

Usage:
    sbatch --gpus=4 pipeline/train_token.sh
"""

import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import typer
import wandb
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from goodfire_core.training.optimizers import EMuon
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from loaders import load_heads
from paths import ARTIFACTS
from pipeline.ref_labels import TokenLabelLookup
from probe.covariance import HeadSpec, MultiHeadCovProbeV2, multihead_loss_v2

DECONFOUNDED = ARTIFACTS / "clinvar_evo2_deconfounded_full"


def load_disruption_specs() -> dict[str, HeadSpec]:
    """Load only disruption heads from heads.json."""
    heads = load_heads()
    return {
        name: HeadSpec(n_classes=info["n_classes"], kind=info["kind"])
        for name, info in heads.items()
        if info["category"] != "effect"
    }


def preload_positions(activations: Path) -> dict[str, torch.Tensor]:
    """Load ALL positions into a dict: sequence_id → [2, 256] int64.

    Every rank needs the full dict, so we read chunks directly (no DDP sharding).
    """
    pos_storage = FilesystemStorage(activations)
    pos_ds = ActivationDataset(pos_storage, "positions", batch_size=512)
    pos_dict: dict[str, torch.Tensor] = {}
    for chunk_id in tqdm(range(pos_ds.num_chunks), desc="Loading positions"):
        chunk = pos_ds.load_chunk(chunk_id)
        for i, sid in enumerate(chunk.sequence_ids):
            pos_dict[sid] = chunk.acts[i]
    logger.info(f"Positions: {len(pos_dict):,} variants")
    return pos_dict


def train(
    name: str,
    activations: Path,
    d_model: int,
    d_hidden: int,
    d_probe: int,
    epochs: int,
    lr: float,
    batch_size: int,
    focal_gamma: float,
    seed: int,
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

    # Heads — disruption only
    specs = load_disruption_specs()
    head_names = tuple(specs.keys())
    specs_tuple = tuple(specs.values())
    n_heads = len(head_names)

    if rank == 0:
        logger.info(f"Disruption heads: {n_heads}")

    # Position + label lookups (CPU, shared across ranks)
    pos_dict = preload_positions(activations)
    lookup = TokenLabelLookup(head_names=head_names)

    # Output
    out_dir = activations / name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        dist.barrier()

    # Activation iterator
    storage = FilesystemStorage(activations)
    per_gpu = max(1, batch_size // world_size)
    act_ds = ActivationDataset(storage, "activations", batch_size=per_gpu)
    iterator = act_ds.training_iterator(device=str(device), n_epochs=epochs, shuffle=True)

    if distributed:
        steps = torch.tensor([iterator.steps_per_epoch], device=device)
        dist.all_reduce(steps, op=dist.ReduceOp.MIN)
        iterator.set_max_steps(int(steps.item()))

    # Probe — bilinear, no sqrtm
    probe = MultiHeadCovProbeV2(
        d_model=d_model, heads=specs, d_hidden=d_hidden, d_probe=d_probe,
        n_sqrtm_iters=0,
    ).to(device)
    optimizer = EMuon(probe.parameters(), lr=lr)

    if distributed:
        probe = DistributedDataParallel(probe, device_ids=[rank])

    if rank == 0:
        wandb.init(project="gfm-probes", name=name, config={
            "training": "token_level",
            "d_model": d_model, "d_hidden": d_hidden, "d_probe": d_probe,
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "focal_gamma": focal_gamma, "n_heads": n_heads, "n_sqrtm_iters": 0,
        })
        logger.info(f"Train: {iterator.steps_per_epoch} steps/epoch, ~{iterator.steps_per_epoch * per_gpu * 256} tokens/epoch")

    # Train
    for epoch in range(epochs):
        probe.train()
        pbar = tqdm(iterator.iter_epoch(), total=iterator.steps_per_epoch,
                    desc=f"Epoch {epoch+1}/{epochs}", disable=rank != 0)

        for batch in pbar:
            B = batch.acts.shape[0]
            raw = batch.acts.float()  # [B, 6291456]

            # Unpack: [B, 2, 3, 256, 4096] → ref view [B, 2, 256, 4096]
            acts = raw.reshape(B, 2, 3, 256, 4096)
            fwd_ref = acts[:, 0, 1]  # [B, 256, 4096]
            bwd_ref = acts[:, 1, 1]  # [B, 256, 4096]
            token_acts = torch.cat([fwd_ref, bwd_ref], dim=-1)  # [B, 256, 8192]

            # Resolve labels: [B, 256, n_heads]
            labels = np.full((B, 256, n_heads), np.nan, dtype=np.float32)
            for b in range(B):
                sid = batch.sequence_ids[b]
                positions = pos_dict.get(sid)
                if positions is None:
                    continue
                chrom = sid.split(":")[0]
                pos_idx = lookup._pos_to_idx.get(chrom)
                chrom_cols = lookup._labels.get(chrom)
                if pos_idx is None or chrom_cols is None:
                    continue
                fwd_pos = positions[0]  # [256] genomic positions
                for t in range(256):
                    row = pos_idx.get(int(fwd_pos[t].item()), -1)
                    if row >= 0:
                        for h in range(n_heads):
                            labels[b, t, h] = float(chrom_cols[h][row])

            # Flatten: [B*256, 8192] acts, [B*256, n_heads] labels
            flat_acts = token_acts.reshape(B * 256, 1, d_model)  # [B*256, 1, 8192] for probe
            flat_labels = torch.from_numpy(labels.reshape(B * 256, n_heads)).to(device, non_blocking=True)

            # Forward + loss
            logits = probe(flat_acts)
            loss = multihead_loss_v2(logits, flat_labels, specs_tuple, focal_gamma=focal_gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")
                wandb.log({"loss": loss.item()})

        if rank == 0:
            raw_p = probe.module if distributed else probe
            raw_p.save_checkpoint(str(out_dir / f"checkpoint_epoch_{epoch+1}.pt"))

    # Save
    if rank == 0:
        raw_p = probe.module if distributed else probe
        raw_p.save_checkpoint(str(out_dir / "weights.pt"))
        (out_dir / "config.json").write_text(json.dumps({
            "training": "token_level",
            "d_model": d_model, "d_hidden": d_hidden, "d_probe": d_probe,
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "focal_gamma": focal_gamma, "n_sqrtm_iters": 0,
            "disruption_heads": list(specs.keys()),
            "n_disruption_heads": n_heads,
        }, indent=2))
        wandb.finish()
        logger.info(f"Saved: {out_dir / 'weights.pt'}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


# ── CLI ──────────────────────────────────────────────────────────────────

app = typer.Typer(help="Train per-token reference probe")


@app.command()
def main(
    name: str = typer.Option("probe_token_v1", help="Output directory name"),
    activations: Path = typer.Option(DECONFOUNDED, help="Activations storage directory"),
    d_model: int = typer.Option(8192),
    d_hidden: int = typer.Option(64),
    d_probe: int = typer.Option(128),
    epochs: int = typer.Option(1),
    lr: float = typer.Option(0.01),
    batch_size: int = typer.Option(64),
    focal_gamma: float = typer.Option(0.0),
    seed: int = typer.Option(42),
) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        os.environ.setdefault("WANDB_DIR", f"/tmp/wandb_{os.environ['SLURM_JOB_ID']}")
    train(**{k: v for k, v in locals().items()})


if __name__ == "__main__":
    app()
