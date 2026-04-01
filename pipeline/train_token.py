"""Train per-token binary probe on reference activations.

Each of the 512 positions per variant (256 downstream + 256 upstream) has
position-specific binary annotations (ATAC-seq, ChIP-seq, cCREs, etc.).
The probe predicts these from individual token activations.

Activation layout: [B, direction=2, view=3, K=256, d=4096]
  ref_same  (view 1) = reference on the selecting strand
  ref_cross (view 2) = reference on the OPPOSITE strand (same positions!)

Token unpacking: for each direction's 256 positions, cat ref_same + ref_cross
to get true bidirectional [fwd, bwd] features at each genomic coordinate.
Result: [B*512, 8192].

Labels: uint8 {0, 1, 255=missing} from token_labels_binary dataset.

Follows the same training_iterator + inject_labels pattern as the sequence
probe (pipeline/train.py). DDP gradient sync is handled by PyTorch DDP
automatically — no manual all_reduce.

Usage:
    sbatch --gpus=4 pipeline/train_token.sh
"""

import json
import os
from pathlib import Path

import torch
import typer
import wandb
from goodfire_core.data.interfaces import TensorActivations
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from goodfire_core.training.optimizers import EMuon
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import torch.distributed as dist
from paths import DECONFOUNDED
from probe.covariance import HeadSpec, MultiHeadCovProbeV2, multihead_loss_v2
from training import ddp_context


def unpack_ref_tokens(raw: torch.Tensor) -> torch.Tensor:
    """[B, 2, 3, 256, 4096] → [B*512, 8192] per-token bidirectional ref.

    ref_cross IS the opposite model direction at the same genomic positions.
    Forward positions: ref_same=fwd, ref_cross=bwd → cat = [fwd, bwd]
    Backward positions: ref_cross=fwd, ref_same=bwd → cat = [fwd, bwd]
    """
    fwd_tokens = torch.cat([raw[:, 0, 1], raw[:, 0, 2]], dim=-1)  # [B, 256, 8192]
    bwd_tokens = torch.cat([raw[:, 1, 2], raw[:, 1, 1]], dim=-1)  # [B, 256, 8192]
    return torch.cat([fwd_tokens, bwd_tokens], dim=1).reshape(-1, 8192)


def load_token_labels(storage: FilesystemStorage) -> dict[str, torch.Tensor]:
    """Pre-load all token labels into memory: {sequence_id → [n_heads, 512] uint8}.

    ~36 GB for 184K variants × 383 heads × 512 positions.
    Same pattern as the sequence probe's label_matrix, just larger.
    """
    ds = ActivationDataset(storage, "token_labels_binary", batch_size=512, include_provenance=True)
    labels: dict[str, torch.Tensor] = {}
    for chunk_id in tqdm(range(ds.num_chunks), desc="Loading token labels"):
        chunk = ds.load_chunk(chunk_id)
        for i, sid in enumerate(chunk.sequence_ids):
            labels[sid] = chunk.acts[i]  # [n_heads, 512] uint8
    logger.info(f"Token labels: {len(labels):,} variants loaded")
    return labels


def train(
    name: str,
    activations: Path,
    d_model: int,
    d_hidden: int,
    d_probe: int,
    lr: float,
    focal_gamma: float,
    batch_size: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)

    with ddp_context() as (device, rank, world_size):
        distributed = world_size > 1

        # Head specs from built dataset
        head_names_path = activations / "token_labels_binary" / "head_names.json"
        head_names = tuple(json.loads(head_names_path.read_text()))
        specs = {name: HeadSpec(n_classes=2, kind="binary") for name in head_names}
        specs_tuple = tuple(specs.values())
        n_heads = len(specs)

        # Pre-load all token labels into memory (~36 GB)
        storage = FilesystemStorage(activations)
        token_labels = load_token_labels(storage)
        missing_labels = torch.full((n_heads, 512), 255, dtype=torch.uint8)

        # Training iterator — handles DDP chunk partitioning, prefetching, batching
        per_gpu = max(1, batch_size // world_size)
        dataset = ActivationDataset(storage, "activations", batch_size=per_gpu)
        iterator = dataset.training_iterator(device=str(device), n_epochs=1)

        # Sync step counts across DDP ranks to prevent deadlocks
        if distributed:
            steps = torch.tensor([iterator.steps_per_epoch], device=device)
            dist.all_reduce(steps, op=dist.ReduceOp.MIN)
            iterator.set_max_steps(int(steps.item()))

        def inject(batch):
            """Unpack tokens and inject per-position labels."""
            tokens = unpack_ref_tokens(batch.acts.float())  # [B*512, 8192]
            B = batch.acts.shape[0]

            labels_list = [token_labels.get(sid, missing_labels) for sid in batch.sequence_ids]
            raw = torch.stack(labels_list)  # [B, n_heads, 512] uint8
            labels = raw.permute(0, 2, 1).reshape(B * 512, n_heads).float()
            labels[labels == 255] = -1  # masked sentinel

            labels = labels.to(tokens.device, non_blocking=True)
            token_sids = [sid for sid in batch.sequence_ids for _ in range(512)]
            return TensorActivations(acts=tokens.unsqueeze(1), labels=labels, sequence_ids=token_sids)

        iterator.add_transform(inject)

        # Probe
        probe = MultiHeadCovProbeV2(
            d_model=d_model, heads=specs, d_hidden=d_hidden, d_probe=d_probe,
            n_sqrtm_iters=0,
        ).to(device)
        optimizer = EMuon(probe.parameters(), lr=lr)

        if distributed:
            probe = DistributedDataParallel(probe, device_ids=[rank])

        out_dir = activations / name
        if rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            wandb.init(project="gfm-probes", name=name, config={
                "training": "token_binary", "d_model": d_model, "d_hidden": d_hidden,
                "d_probe": d_probe, "lr": lr, "focal_gamma": focal_gamma,
                "batch_size": batch_size, "n_heads": n_heads, "n_sqrtm_iters": 0,
            })
            logger.info(f"Binary heads: {n_heads}, steps: {iterator.steps_per_epoch}, batch/gpu: {per_gpu}")

        if distributed:
            dist.barrier()

        # Train — single epoch
        probe.train()
        pbar = tqdm(iterator.iter_epoch(), total=iterator.steps_per_epoch,
                     desc="Training", disable=rank != 0)

        for batch in pbar:
            if batch.batch_size == 0 or batch.labels is None:
                continue

            logits = probe(batch.acts)
            loss = multihead_loss_v2(logits, batch.labels, specs_tuple, focal_gamma=focal_gamma)

            if not loss.requires_grad:
                continue  # all labels masked in this batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")
                wandb.log({"loss": loss.item()})

        pbar.close()

        if rank == 0:
            raw_p = probe.module if distributed else probe
            raw_p.save_checkpoint(str(out_dir / "weights.pt"))
            (out_dir / "config.json").write_text(json.dumps({
                "training": "token_binary", "d_model": d_model, "d_hidden": d_hidden,
                "d_probe": d_probe, "lr": lr, "focal_gamma": focal_gamma,
                "batch_size": batch_size, "n_sqrtm_iters": 0,
                "heads": list(head_names), "n_heads": n_heads,
            }, indent=2))
            wandb.finish()
            logger.info(f"Saved: {out_dir / 'weights.pt'}")


# ── CLI ──────────────────────────────────────────────────────────────────

app = typer.Typer(help="Train per-token binary probe")


@app.command()
def main(
    name: str = typer.Option("probe_token_v1", help="Output directory name"),
    activations: Path = typer.Option(DECONFOUNDED, help="Activations storage directory"),
    d_model: int = typer.Option(8192),
    d_hidden: int = typer.Option(64),
    d_probe: int = typer.Option(128),
    lr: float = typer.Option(0.01),
    focal_gamma: float = typer.Option(0.0),
    batch_size: int = typer.Option(2, help="Variants per batch (each = 512 tokens)"),
    seed: int = typer.Option(42),
) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        os.environ.setdefault("WANDB_DIR", f"/tmp/wandb_{os.environ['SLURM_JOB_ID']}")
    train(**{k: v for k, v in locals().items()})


if __name__ == "__main__":
    app()
