"""Train v11 multihead covariance probe: reference pretrain + variant finetune.

Phase 1 (pretrain): Train local+global ref heads on 190M single-token SAE
  activations. The covariance probe degenerates to a bilinear form on single
  tokens. Labels are resolved at batch time from token annotation parquets.

Phase 2 (finetune): Train effect heads on variant diff view, continue global
  ref heads on variant ref view, freeze local ref heads.

Usage:
    ACTS=/mnt/polished-lake/artifacts/.../clinvar_evo2_deconfounded_full

    # Phase 1
    sbatch --gpus=4 --time=6:00:00 --wrap \\
        "uv run torchrun --nproc-per-node=4 scripts/train_probe_v11.py \\
         --phase pretrain --activations $ACTS"

    # Phase 2
    sbatch --gpus=4 --time=4:00:00 --wrap \\
        "uv run torchrun --nproc-per-node=4 scripts/train_probe_v11.py \\
         --phase finetune --activations $ACTS \\
         --checkpoint $ACTS/probe_v11/pretrain.pt"
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import wandb
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from goodfire_core.training.optimizers import EMuon
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from training import gene_split
from probe.covariance import HeadSpec, MultiHeadCovProbeV2, multihead_loss
from loaders import load_metadata
from pipeline.extract import unified_diff, unified_ref
from pipeline.ref_labels import RefChunkLoader

# ── Paths ─────────────────────────────────────────────────────────────────

ARTIFACTS = Path("/mnt/polished-lake/artifacts/fellows-shared/life-sciences/genomics/mendelian")
DECONFOUNDED = ARTIFACTS / "clinvar_evo2_deconfounded_full"
ANNOTATIONS = DECONFOUNDED / "annotations.feather"
HEAD_CATEGORIES = DECONFOUNDED / "head_categories.json"
SAE_DIR = ARTIFACTS / "mendelian_saes/bidirectional_saes_v1/sae_datasets/train"


# ── Head setup ────────────────────────────────────────────────────────────


def load_head_specs() -> tuple[dict[str, HeadSpec], dict[str, HeadSpec], dict[str, HeadSpec]]:
    """Load head definitions split into (effect, local_ref, global_ref)."""
    categories = json.loads(HEAD_CATEGORIES.read_text())
    def to_specs(d): return {n: HeadSpec(n_classes=v["n_classes"], kind=v["kind"]) for n, v in d.items()}
    return to_specs(categories["effect"]), to_specs(categories["local_ref"]), to_specs(categories["global_ref"])


# ── Phase 1: Reference pretraining ───────────────────────────────────────


def pretrain(args, device, rank, world_size):
    """Pretrain local+global ref heads on single-token SAE activations."""
    effect_specs, local_specs, global_specs = load_head_specs()
    ref_specs = {**local_specs, **global_specs}
    all_specs = {**ref_specs, **effect_specs}

    head_names = list(all_specs.keys())
    ref_names = tuple(ref_specs.keys())
    specs_tuple = tuple(all_specs.values())

    logger.info(f"Heads: {len(local_specs)} local + {len(global_specs)} global + {len(effect_specs)} effect (masked)")

    # Mask template: effect heads get -1 (categorical) or NaN (continuous)
    missing = torch.full((len(head_names),), float("nan"), dtype=torch.float32)
    for i, spec in enumerate(specs_tuple):
        if spec.kind != "continuous":
            missing[i] = -1

    effect_indices = [i for i, name in enumerate(head_names) if name in effect_specs]

    # Chunk loader: streams SAE activations with position-resolved labels
    loader = RefChunkLoader(
        sae_dir=SAE_DIR,
        head_names=ref_names,
        batch_size=max(1, args.batch_size // world_size),
        device=str(device),
    )

    # Map ref head labels → full head order (effect heads stay masked)
    ref_to_full = {}
    for ri, ref_name in enumerate(ref_names):
        for fi, full_name in enumerate(head_names):
            if ref_name == full_name:
                ref_to_full[ri] = fi
                break

    # Probe
    probe = MultiHeadCovProbeV2(
        d_model=args.d_model, heads=all_specs,
        d_hidden=args.d_hidden, d_probe=args.d_probe,
    ).to(device)
    optimizer = EMuon(probe.parameters(), lr=args.lr)

    distributed = world_size > 1
    if distributed:
        probe = DistributedDataParallel(probe, device_ids=[rank])

    out_dir = args.activations / "probe_v11"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Train
    probe.train()
    step = 0
    for acts, ref_labels, valid in tqdm(loader.iter_epoch(), disable=rank != 0, desc="pretrain"):
        # Build full label vector: ref labels in position, effect heads masked
        B = acts.shape[0]
        labels = missing.unsqueeze(0).expand(B, -1).clone().to(device, non_blocking=True)
        for ri, fi in ref_to_full.items():
            labels[:, fi] = ref_labels[:, ri]

        # Mask effect heads
        for i in effect_indices:
            labels[:, i] = missing[i]

        # Forward: single token → covariance degenerates to bilinear
        # acts: [B, 8192], probe expects [B, K, d] so unsqueeze K=1
        logits = probe(acts.unsqueeze(1))
        loss = multihead_loss(logits, labels, specs_tuple, focal_gamma=args.focal_gamma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        if rank == 0 and step % 100 == 0:
            logger.info(f"Step {step}: loss={loss.item():.4f}")

    # Save
    if rank == 0:
        raw_probe = probe.module if distributed else probe
        raw_probe.save_checkpoint(str(out_dir / "pretrain.pt"))
        (out_dir / "pretrain_config.json").write_text(json.dumps({
            "phase": "pretrain",
            "d_model": args.d_model,
            "d_hidden": args.d_hidden,
            "d_probe": args.d_probe,
            "lr": args.lr,
            "focal_gamma": args.focal_gamma,
            "batch_size": args.batch_size,
            "n_ref_heads": len(ref_specs),
            "n_local": len(local_specs),
            "n_global": len(global_specs),
            "total_steps": step,
        }, indent=2))
        logger.info(f"Saved: {out_dir / 'pretrain.pt'} ({step} steps)")


# ── Phase 2: Variant finetuning ──────────────────────────────────────────


def finetune(args, device, rank, world_size):
    """Finetune on variants: effect heads (diff), global ref heads (ref view)."""
    effect_specs, local_specs, global_specs = load_head_specs()
    all_specs = {**local_specs, **global_specs, **effect_specs}

    head_names = list(all_specs.keys())
    local_indices = [i for i, name in enumerate(head_names) if name in local_specs]
    global_indices = [i for i, name in enumerate(head_names) if name in global_specs]
    effect_indices = [i for i, name in enumerate(head_names) if name in effect_specs]
    specs_tuple = tuple(all_specs.values())

    # Load annotations + gene split
    annotations = pl.read_ipc(ANNOTATIONS)
    manifest = load_metadata(args.preset)
    train_df, test_df = gene_split(manifest, test_size=args.test_size, seed=args.seed)
    train_ids = train_df["variant_id"].to_list()

    # Build label matrix in head_names order
    label_cols = []
    for name in head_names:
        if name in annotations.columns:
            label_cols.append(annotations[name].to_numpy().astype("float32"))
        else:
            spec = all_specs[name]
            fill = float("nan") if spec.kind == "continuous" else -1.0
            label_cols.append(np.full(annotations.height, fill, dtype=np.float32))
    label_matrix = torch.from_numpy(np.stack(label_cols, axis=1))
    id_to_idx = {vid: i for i, vid in enumerate(annotations["variant_id"].to_list())}

    # Missing label template
    missing = torch.full((len(head_names),), float("nan"), dtype=torch.float32)
    for i, spec in enumerate(specs_tuple):
        if spec.kind != "continuous":
            missing[i] = -1

    # Probe: load pretrain checkpoint if available
    probe = MultiHeadCovProbeV2(
        d_model=args.d_model, heads=all_specs,
        d_hidden=args.d_hidden, d_probe=args.d_probe,
    )
    if args.checkpoint:
        ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        probe.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info(f"Loaded pretrain: {args.checkpoint}")
    probe = probe.to(device)

    # Freeze local ref heads
    local_prefixes = tuple(f"heads.{name}." for name in local_specs)
    for param_name, param in probe.named_parameters():
        if param_name.startswith(local_prefixes):
            param.requires_grad_(False)

    optimizer = EMuon([p for p in probe.parameters() if p.requires_grad], lr=args.lr)

    distributed = world_size > 1
    if distributed:
        probe = DistributedDataParallel(probe, device_ids=[rank])

    # Variant iterator
    storage = FilesystemStorage(args.activations)
    per_gpu = max(1, args.batch_size // world_size)
    dataset = ActivationDataset(storage, "activations", batch_size=per_gpu)
    var_iter = dataset.training_iterator(
        device=str(device), n_epochs=args.epochs, sequence_ids=train_ids,
    )

    out_dir = args.activations / "probe_v11"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        pl.concat([
            train_df.select("variant_id", "gene_name").with_columns(pl.lit("train").alias("split")),
            test_df.select("variant_id", "gene_name").with_columns(pl.lit("test").alias("split")),
        ]).write_ipc(out_dir / "split.feather")

    for epoch in range(args.epochs):
        probe.train()
        pbar = tqdm(var_iter.iter_epoch(), total=var_iter.steps_per_epoch,
                    desc=f"finetune e{epoch+1}", disable=rank != 0)

        for batch in pbar:
            raw = batch.acts
            B = raw.shape[0]

            # Look up labels
            indices = torch.tensor([id_to_idx.get(sid, -1) for sid in batch.sequence_ids], dtype=torch.long)
            known = indices >= 0
            labels = missing.unsqueeze(0).expand(B, -1).clone()
            if known.any():
                labels[known] = label_matrix[indices[known]]
            labels = labels.to(device, non_blocking=True)

            # Mask by view
            ref_labels = labels.clone()
            diff_labels = labels.clone()
            for i in effect_indices:
                ref_labels[:, i] = missing[i]
            for i in local_indices + global_indices:
                diff_labels[:, i] = missing[i]

            # Pass 1: diff → effect heads
            diff_logits = probe(unified_diff(raw).float())
            loss_diff = multihead_loss(diff_logits, diff_labels, specs_tuple, focal_gamma=args.focal_gamma)

            # Pass 2: ref → global ref heads (locals frozen)
            ref_logits = probe(unified_ref(raw).float())
            loss_ref = multihead_loss(ref_logits, ref_labels, specs_tuple, focal_gamma=args.focal_gamma)

            loss = loss_diff + loss_ref
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}", diff=f"{loss_diff.item():.3f}", ref=f"{loss_ref.item():.3f}")

    # Save
    if rank == 0:
        raw_probe = probe.module if distributed else probe
        raw_probe.save_checkpoint(str(out_dir / "weights.pt"))
        (out_dir / "config.json").write_text(json.dumps({
            "phase": "finetune",
            "preset": args.preset,
            "d_model": args.d_model,
            "d_hidden": args.d_hidden,
            "d_probe": args.d_probe,
            "lr": args.lr,
            "epochs": args.epochs,
            "focal_gamma": args.focal_gamma,
            "n_effect": len(effect_specs),
            "n_local_ref": len(local_specs),
            "n_global_ref": len(global_specs),
            "pretrain": str(args.checkpoint) if args.checkpoint else None,
            "n_train": train_df.height,
            "n_test": test_df.height,
        }, indent=2))
        logger.info(f"Saved: {out_dir / 'weights.pt'}")


# ── CLI ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=("pretrain", "finetune"))
    parser.add_argument("--name", default="probe-v11", help="W&B run name")
    parser.add_argument("--activations", type=Path, default=DECONFOUNDED)
    parser.add_argument("--checkpoint", type=Path, help="Pretrain checkpoint for finetune")
    parser.add_argument("--preset", default="deconfounded-full")
    parser.add_argument("--d-model", type=int, default=8192)
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--d-probe", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--focal-gamma", type=float, default=3.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    distributed = dist.is_available() and "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda")

    if os.environ.get("SLURM_JOB_ID"):
        os.environ.setdefault("WANDB_DIR", f"/tmp/wandb_{os.environ['SLURM_JOB_ID']}")
    if rank == 0:
        wandb.init(project="gfm-probes", name=args.name, config=vars(args))

    if args.phase == "pretrain":
        pretrain(args, device, rank, world_size)
    else:
        finetune(args, device, rank, world_size)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
