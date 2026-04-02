"""Train per-token probe on reference activations.

Supports binary and continuous heads. Labels loaded from safetensors (mmap'd,
per-head keyed). Each DDP rank only touches its slice of variants via the OS
page cache — no explicit sharding needed.

Requires token_labels.safetensors + token_labels.json from build_token_labels.py.

Usage:
    sbatch --gpus=8 pipeline/train_token.sh --name probe_token_v2
"""

import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import typer
from goodfire_core.data.interfaces import TensorActivations
from goodfire_core.storage import FilesystemStorage
from goodfire_core.training.optimizers import EMuon
from loguru import logger
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from paths import DECONFOUNDED
from probe.covariance import HeadSpec, MultiHeadCovProbeV2, multihead_loss_v2
from training import (
    ddp_context,
    fire_callbacks,
    local_activation_dataset,
    setup_training_callbacks,
)


def unpack_ref_tokens(raw: torch.Tensor) -> torch.Tensor:
    """[B, 2, 3, 256, 4096] → [B*512, 1, 8192] per-token bidirectional ref."""
    fwd_tokens = torch.cat([raw[:, 0, 1], raw[:, 0, 2]], dim=-1)
    bwd_tokens = torch.cat([raw[:, 1, 2], raw[:, 1, 1]], dim=-1)
    return torch.cat([fwd_tokens, bwd_tokens], dim=1).reshape(-1, 1, 8192)


# ── Label loading ─────────────────────────────────────────────────────


def load_token_labels(
    activations: Path,
) -> tuple[
    dict[str, int],  # id_to_idx
    list[torch.Tensor],  # per-head label tensors [N, 512] (mmap'd)
    list[str],  # head names (ordered: binary first, then continuous)
    list[HeadSpec],  # head specs (same order)
]:
    """Load token labels from safetensors + JSON sidecar.

    Returns mmap'd tensors — no memory cost until rows are accessed.
    """
    labels_path = activations / "token_labels.safetensors"
    meta_path = activations / "token_labels.json"

    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Run: uv run python pipeline/build_token_labels.py {activations}"
        )

    meta = json.loads(meta_path.read_text())
    id_to_idx = {vid: i for i, vid in enumerate(meta["ids"])}

    binary_heads = meta["binary_heads"]
    continuous_heads = meta.get("continuous_heads", [])
    n_continuous_bins = 16

    # Build ordered head names + specs
    head_names: list[str] = []
    head_specs: list[HeadSpec] = []
    for h in binary_heads:
        head_names.append(h)
        head_specs.append(HeadSpec(n_classes=2, kind="binary"))
    # Each continuous head's init loss is log(n_bins). Scale so it matches
    # binary init loss log(2), keeping the per-head average at log(2).
    continuous_weight = math.log(2) / math.log(n_continuous_bins)
    for h in continuous_heads:
        head_names.append(h)
        head_specs.append(HeadSpec(n_classes=n_continuous_bins, kind="continuous", weight=continuous_weight))

    # Load tensors via mmap
    # safe_open mmaps the file; we hold a reference to keep it alive for training.
    # Tensors returned by get_tensor() are views into the mmap.
    label_file = safe_open(str(labels_path), framework="pt", device="cpu")
    label_tensors = [label_file.get_tensor(h) for h in head_names]  # each [N, 512]

    return id_to_idx, label_tensors, head_names, head_specs, label_file


# ── Training ──────────────────────────────────────────────────────────


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

        # Load labels (mmap'd — each rank opens the file, OS shares pages)
        id_to_idx, label_tensors, head_names, head_specs, _label_file = load_token_labels(activations)
        specs = dict(zip(head_names, head_specs))
        specs_tuple = tuple(head_specs)
        n_heads = len(head_names)
        n_binary = sum(1 for s in head_specs if s.kind == "binary")
        n_continuous = n_heads - n_binary

        if rank == 0:
            logger.info(f"Heads: {n_binary} binary + {n_continuous} continuous = {n_heads}")
            total_gb = sum(t.nbytes for t in label_tensors) / 1e9
            logger.info(f"Labels: {total_gb:.1f} GB safetensors (mmap'd)")

        # Output
        out_dir = activations / name
        if rank == 0:
            out_dir.mkdir(parents=True, exist_ok=True)

        if distributed:
            dist.barrier()

        # Iterator for raw activations
        storage = FilesystemStorage(activations)
        per_gpu = max(1, batch_size // world_size)
        dataset = local_activation_dataset(storage, "activations", batch_size=per_gpu)
        iterator = dataset.training_iterator(device=str(device), n_epochs=epochs)

        if distributed:
            steps = torch.tensor([iterator.steps_per_epoch], device=device)
            dist.all_reduce(steps, op=dist.ReduceOp.MIN)
            iterator.set_max_steps(int(steps.item()))

        # Inject per-token labels from safetensors
        def inject_labels(batch):
            tokens = unpack_ref_tokens(batch.acts.float())  # [B*512, 1, 8192]
            B = batch.acts.shape[0]

            indices = [id_to_idx.get(sid, -1) for sid in batch.sequence_ids]
            idx_t = torch.tensor(indices, dtype=torch.long)
            known = idx_t >= 0
            known_idx = idx_t[known]

            # Build [B, 512, n_heads] label matrix from per-head tensors
            labels = torch.full((B, 512, n_heads), -1.0)  # default: masked

            if known.any():
                for h_idx, (tensor, spec) in enumerate(zip(label_tensors, head_specs)):
                    head_data = tensor[known_idx]  # [n_known, 512], mmap'd read
                    if spec.kind == "binary":
                        col = head_data.float()
                        col[col == 255] = -1  # missing → masked
                        labels[known, :, h_idx] = col
                    else:
                        col = head_data.float()
                        # NaN stays NaN (continuous missing marker)
                        labels[known, :, h_idx] = col

            labels = labels.reshape(B * 512, n_heads).to(tokens.device, non_blocking=True)
            return TensorActivations(acts=tokens, labels=labels, sequence_ids=None)

        iterator.add_transform(inject_labels)

        # Probe
        probe = MultiHeadCovProbeV2(
            d_model=d_model,
            heads=specs,
            d_hidden=d_hidden,
            d_probe=d_probe,
            n_sqrtm_iters=0,
        ).to(device)
        optimizer = EMuon(probe.parameters(), lr=lr)

        if distributed:
            probe = DistributedDataParallel(probe, device_ids=[rank])

        config = {
            "training": "token",
            "d_model": d_model,
            "d_hidden": d_hidden,
            "d_probe": d_probe,
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "focal_gamma": focal_gamma,
            "n_heads": n_heads,
            "n_sqrtm_iters": 0,
            "n_binary": n_binary,
            "n_continuous": n_continuous,
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
            logger.info(f"Train: {len(id_to_idx):,} variants, {iterator.steps_per_epoch} steps/epoch")

        # Train — single pass (no dual view)
        raw_p = probe.module if distributed else probe
        step = 0
        for epoch in range(epochs):
            probe.train()
            fire_callbacks(callbacks, "on_epoch_begin", epoch=epoch)
            pbar = tqdm(
                iterator.iter_epoch(),
                total=iterator.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{epochs}",
                disable=rank != 0,
            )
            epoch_loss = 0.0
            epoch_samples = 0

            for batch in pbar:
                if batch.batch_size == 0 or batch.labels is None:
                    continue

                logits = probe(batch.acts)
                loss = multihead_loss_v2(logits, batch.labels, specs_tuple, focal_gamma=focal_gamma)

                if not loss.requires_grad:
                    continue

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1000.0).item()
                optimizer.step()

                step += 1
                epoch_loss += loss.item() * batch.batch_size
                epoch_samples += batch.batch_size

                if rank == 0:
                    pbar.set_postfix(loss=f"{loss.item():.3f}", gn=f"{grad_norm:.1f}")
                    fire_callbacks(
                        callbacks,
                        "on_step_end",
                        step=step,
                        loss=loss.item(),
                        grad_norm=grad_norm,
                        batch_size=batch.batch_size,
                        log_frequency=1,
                    )

            fire_callbacks(
                callbacks, "on_epoch_end", model=raw_p, epoch=epoch, train_loss=epoch_loss / max(epoch_samples, 1)
            )

        # Save final weights + config
        if rank == 0:
            raw_p.save_checkpoint(str(out_dir / "weights.pt"))
            config["heads"] = list(head_names)
            (out_dir / "config.json").write_text(json.dumps(config, indent=2))
            logger.info(f"Saved: {out_dir / 'weights.pt'}")

        fire_callbacks(callbacks, "on_train_end", model=raw_p, epoch=epochs - 1)


# ── CLI ──────────────────────────────────────────────────────────────────

app = typer.Typer(help="Train per-token probe")


@app.command()
def main(
    name: str = typer.Option("probe_token_v2", help="Output directory name"),
    activations: Path = typer.Option(DECONFOUNDED, help="Activations storage directory"),
    d_model: int = typer.Option(8192),
    d_hidden: int = typer.Option(64),
    d_probe: int = typer.Option(128),
    epochs: int = typer.Option(1),
    lr: float = typer.Option(0.01),
    batch_size: int = typer.Option(64, help="Total variants per step (8 per GPU × 8 GPUs; each variant = 512 tokens)"),
    focal_gamma: float = typer.Option(0.0),
    seed: int = typer.Option(42),
) -> None:
    if os.environ.get("SLURM_JOB_ID"):
        os.environ.setdefault("WANDB_DIR", f"/tmp/wandb_{os.environ['SLURM_JOB_ID']}")
    train(**{k: v for k, v in locals().items()})


if __name__ == "__main__":
    app()
