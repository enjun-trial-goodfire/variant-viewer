"""Train per-token binary probe on reference activations.

Minimal diff from pipeline/train.py. Same batch_size, same training_iterator,
same DDP. Only changes: heads (binary only), labels (per-token from
token_labels_binary), forward (unpack tokens), single-pass (no dual view).

Usage:
    sbatch --gpus=8 pipeline/train_token.sh --name probe_token_v1
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import typer
import wandb
from goodfire_core.data.interfaces import TensorActivations
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from goodfire_core.training.optimizers import EMuon
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from paths import DECONFOUNDED
from probe.covariance import HeadSpec, MultiHeadCovProbeV2, multihead_loss_v2
from training import ddp_context


# ── CHANGE 1: token unpacking (replaces unified_diff/unified_ref) ───────

def unpack_ref_tokens(raw: torch.Tensor) -> torch.Tensor:
    """[B, 2, 3, 256, 4096] → [B*512, 1, 8192] per-token bidirectional ref."""
    fwd_tokens = torch.cat([raw[:, 0, 1], raw[:, 0, 2]], dim=-1)
    bwd_tokens = torch.cat([raw[:, 1, 2], raw[:, 1, 1]], dim=-1)
    return torch.cat([fwd_tokens, bwd_tokens], dim=1).reshape(-1, 1, 8192)


# ── Training (identical structure to train.py) ──────────────────────────


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

    with ddp_context() as (device, rank, world_size):
        distributed = world_size > 1

        # CHANGE 2: heads — binary disruption only
        head_names = tuple(json.loads(
            (activations / "token_labels_binary" / "head_names.json").read_text()
        ))
        specs = {name: HeadSpec(n_classes=2, kind="binary") for name in head_names}
        specs_tuple = tuple(specs.values())
        n_heads = len(specs)

        if rank == 0:
            logger.info(f"Heads: {n_heads} binary disruption")

        # CHANGE 3: labels — contiguous tensor from token_labels_binary
        storage = FilesystemStorage(activations)
        labels_ds = ActivationDataset(storage, "token_labels_binary", batch_size=512, include_provenance=True)
        all_ids: list[str] = []
        all_labels: list[torch.Tensor] = []
        for chunk_id in tqdm(range(labels_ds.num_chunks), desc="Loading token labels", disable=rank != 0):
            chunk = labels_ds.load_chunk(chunk_id)
            all_ids.extend(chunk.sequence_ids)
            all_labels.append(chunk.acts)
        label_matrix = torch.cat(all_labels, dim=0)  # [N, n_heads, 512] uint8
        id_to_idx = {vid: i for i, vid in enumerate(all_ids)}
        del all_labels, all_ids

        if rank == 0:
            logger.info(f"Token labels: {label_matrix.shape}, {label_matrix.nbytes / 1e9:.1f} GB")

        missing_label = torch.full((n_heads, 512), 255, dtype=torch.uint8)

        # Output
        out_dir = activations / name
        if rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)

        if distributed:
            dist.barrier()

        # Iterator — SAME as sequence probe
        per_gpu = max(1, batch_size // world_size)
        dataset = ActivationDataset(storage, "activations", batch_size=per_gpu)
        iterator = dataset.training_iterator(device=str(device), n_epochs=epochs)

        if distributed:
            steps = torch.tensor([iterator.steps_per_epoch], device=device)
            dist.all_reduce(steps, op=dist.ReduceOp.MIN)
            iterator.set_max_steps(int(steps.item()))

        # CHANGE 4: inject token labels (same pattern, different reshape)
        def inject_labels(batch):
            tokens = unpack_ref_tokens(batch.acts.float())  # [B*512, 1, 8192]
            B = batch.acts.shape[0]

            indices = [id_to_idx.get(sid, -1) for sid in batch.sequence_ids]
            idx_t = torch.tensor(indices, dtype=torch.long)
            known = idx_t >= 0

            raw_labels = missing_label.unsqueeze(0).expand(B, -1, -1).clone()
            if known.any():
                raw_labels[known] = label_matrix[idx_t[known]]

            labels = raw_labels.permute(0, 2, 1).reshape(B * 512, n_heads).float()
            labels[labels == 255] = -1
            labels = labels.to(tokens.device, non_blocking=True)
            return TensorActivations(acts=tokens, labels=labels, sequence_ids=None)

        iterator.add_transform(inject_labels)

        # Probe — SAME architecture, n_sqrtm_iters=0
        probe = MultiHeadCovProbeV2(
            d_model=d_model, heads=specs, d_hidden=d_hidden, d_probe=d_probe,
            n_sqrtm_iters=0,
        ).to(device)
        optimizer = EMuon(probe.parameters(), lr=lr)

        if distributed:
            probe = DistributedDataParallel(probe, device_ids=[rank])

        if rank == 0:
            wandb.init(project="gfm-probes", name=name, config={
                "training": "token_binary", "d_model": d_model, "d_hidden": d_hidden,
                "d_probe": d_probe, "lr": lr, "epochs": epochs, "batch_size": batch_size,
                "focal_gamma": focal_gamma, "n_heads": n_heads, "n_sqrtm_iters": 0,
            })
            logger.info(f"Train: {label_matrix.shape[0]:,} variants, {iterator.steps_per_epoch} steps/epoch")

        # Train — SAME loop, single pass instead of dual
        for epoch in range(epochs):
            probe.train()
            pbar = tqdm(iterator.iter_epoch(), total=iterator.steps_per_epoch,
                        desc=f"Epoch {epoch+1}/{epochs}", disable=rank != 0)

            for batch in pbar:
                if batch.batch_size == 0 or batch.labels is None:
                    continue

                # CHANGE 5: single forward pass (no dual view)
                logits = probe(batch.acts)
                loss = multihead_loss_v2(logits, batch.labels, specs_tuple, focal_gamma=focal_gamma)

                if not loss.requires_grad:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if rank == 0:
                    pbar.set_postfix(loss=f"{loss.item():.3f}")
                    wandb.log({"loss": loss.item()})

        # Save
        if rank == 0:
            raw_p = probe.module if distributed else probe
            raw_p.save_checkpoint(str(out_dir / "weights.pt"))
            (out_dir / "config.json").write_text(json.dumps({
                "training": "token_binary", "d_model": d_model, "d_hidden": d_hidden,
                "d_probe": d_probe, "lr": lr, "epochs": epochs, "batch_size": batch_size,
                "focal_gamma": focal_gamma, "n_sqrtm_iters": 0,
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
    epochs: int = typer.Option(1),
    lr: float = typer.Option(0.01),
    batch_size: int = typer.Option(256),
    focal_gamma: float = typer.Option(0.0),
    seed: int = typer.Option(42),
) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        os.environ.setdefault("WANDB_DIR", f"/tmp/wandb_{os.environ['SLURM_JOB_ID']}")
    train(**{k: v for k, v in locals().items()})


if __name__ == "__main__":
    app()
