"""Train per-token reference probe by co-iterating activations + token_labels.

Streams activations and pre-generated token_labels chunk by chunk (same chunk
boundaries, same sequence IDs). Each variant's [2, 3, 256, 4096] activation
tensor is unpacked to [256, 8192] per-token vectors. Labels come from the
aligned token_labels dataset [443, 256] per variant.

Only disruption (reference) heads. Probe with n_sqrtm_iters=0 (bilinear).
Summing token logits at inference = sequence-level prediction (linearity).

Usage:
    sbatch --gpus=4 pipeline/train_token.sh
"""

import json
import os
from datetime import timedelta
from pathlib import Path

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

    # Datasets
    storage = FilesystemStorage(activations)
    acts_ds = ActivationDataset(storage, "activations", batch_size=1, include_provenance=True)
    labels_ds = ActivationDataset(storage, "token_labels", batch_size=1, include_provenance=True)

    assert acts_ds.num_chunks == labels_ds.num_chunks, (
        f"Chunk count mismatch: activations={acts_ds.num_chunks}, token_labels={labels_ds.num_chunks}"
    )
    n_chunks = acts_ds.num_chunks

    # DDP: partition chunks across ranks
    rank_chunks = list(range(rank, n_chunks, world_size))
    steps_per_epoch = len(rank_chunks)

    if rank == 0:
        logger.info(f"Chunks: {n_chunks} total, {steps_per_epoch} per rank, ~{steps_per_epoch * 39 * 256} tokens/epoch")

    # Probe — bilinear, no sqrtm
    probe = MultiHeadCovProbeV2(
        d_model=d_model, heads=specs, d_hidden=d_hidden, d_probe=d_probe,
        n_sqrtm_iters=0,
    ).to(device)
    optimizer = EMuon(probe.parameters(), lr=lr)

    if distributed:
        probe = DistributedDataParallel(probe, device_ids=[rank])

    # Output
    out_dir = activations / name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(project="gfm-probes", name=name, config={
            "training": "token_level",
            "d_model": d_model, "d_hidden": d_hidden, "d_probe": d_probe,
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "focal_gamma": focal_gamma, "n_heads": n_heads, "n_sqrtm_iters": 0,
        })

    if distributed:
        dist.barrier()

    # Train
    for epoch in range(epochs):
        probe.train()

        # Shuffle chunk order per epoch
        g = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(len(rank_chunks), generator=g)
        epoch_chunks = [rank_chunks[i] for i in perm]

        pbar = tqdm(epoch_chunks, desc=f"Epoch {epoch+1}/{epochs}", disable=rank != 0)

        for chunk_id in pbar:
            acts_chunk = acts_ds.load_chunk(chunk_id)
            labels_chunk = labels_ds.load_chunk(chunk_id)

            B = acts_chunk.acts.shape[0]

            # Unpack on CPU first to avoid putting full [B, 2, 3, 256, 4096] on GPU
            raw = acts_chunk.acts.float()  # CPU
            acts = raw.reshape(B, 2, 3, 256, 4096)
            fwd_ref = acts[:, 0, 1]  # [B, 256, 4096] CPU
            bwd_ref = acts[:, 1, 1]  # [B, 256, 4096] CPU
            token_acts = torch.cat([fwd_ref, bwd_ref], dim=-1)  # [B, 256, 8192] CPU

            # Labels on CPU: [B, n_heads, 256] → [B, 256, n_heads]
            labels = labels_chunk.acts.float().permute(0, 2, 1)  # CPU

            # Flatten and move only what's needed to GPU
            n_tokens = B * 256
            flat_acts = token_acts.reshape(n_tokens, 1, d_model).to(device, non_blocking=True)
            flat_labels = labels.reshape(n_tokens, n_heads).to(device, non_blocking=True)

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
