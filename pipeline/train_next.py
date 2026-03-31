"""Train multihead covariance probe: reference pretrain + variant finetune.

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

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import typer
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from goodfire_core.training.optimizers import EMuon
from loguru import logger
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import wandb
from loaders import load_heads, load_variants

# ── Paths ─────────────────────────────────────────────────────────────────
from paths import ARTIFACTS
from pipeline.extract import unified_diff, unified_ref
from pipeline.ref_labels import RefChunkLoader
from probe.covariance import HeadSpec, MultiHeadCovProbeV2
from probe.covariance import multihead_loss_v2 as multihead_loss
from training import gene_split

DECONFOUNDED = ARTIFACTS / "clinvar_evo2_deconfounded_full"
SAE_DIR = ARTIFACTS / "mendelian_saes/bidirectional_saes_v1/sae_datasets/train"


# ── Head setup ────────────────────────────────────────────────────────────


def load_head_specs() -> tuple[dict[str, HeadSpec], dict[str, HeadSpec]]:
    """Load head definitions split into (disruption, effect)."""
    heads = load_heads()
    disruption, effect = {}, {}
    for name, info in heads.items():
        spec = HeadSpec(n_classes=info["n_classes"], kind=info["kind"])
        if info["category"] == "effect":
            effect[name] = spec
        else:
            disruption[name] = spec
    return disruption, effect


# ── Phase 1: Reference pretraining ───────────────────────────────────────


def pretrain(args, device, rank, world_size):
    """Pretrain local+global ref heads on single-token SAE activations."""
    disruption_specs, effect_specs = load_head_specs()
    all_specs = {**disruption_specs, **effect_specs}

    head_names = list(all_specs.keys())
    ref_names = tuple(disruption_specs.keys())
    specs_tuple = tuple(all_specs.values())

    logger.info(f"Heads: {len(disruption_specs)} disruption + {len(effect_specs)} effect (masked)")

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
    for acts, ref_labels, _valid in tqdm(loader.iter_epoch(), disable=rank != 0, desc="pretrain"):
        # Build full label vector: ref labels in position, effect heads masked
        n = acts.shape[0]
        labels = missing.unsqueeze(0).expand(n, -1).clone().to(device, non_blocking=True)
        for ri, fi in ref_to_full.items():
            labels[:, fi] = ref_labels[:, ri]

        # Mask effect heads
        for i in effect_indices:
            labels[:, i] = missing[i]

        # Forward: single token → covariance degenerates to bilinear
        # acts: [B, 8192], probe expects [B, K, d] so unsqueeze K=1
        logits = probe(acts.unsqueeze(1))
        loss = multihead_loss(logits, labels, specs_tuple, focal_gamma=args.focal_gamma, class_balance=True)

        optimizer.zero_grad()
        loss.backward()

        # Log gradient norm before stepping
        if rank == 0:
            raw_p = probe.module if distributed else probe
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_p.parameters(), max_norm=float("inf"))
            wandb.log({
                "pretrain/loss": loss.item(),
                "pretrain/grad_norm": grad_norm.item(),
                "pretrain/step": step,
                "pretrain/batch_size": n,
            })

        optimizer.step()

        step += 1
        if rank == 0 and step % 100 == 0:
            logger.info(f"Step {step}: loss={loss.item():.4f}")

    # Save + upload artifact
    if rank == 0:
        raw_probe = probe.module if distributed else probe
        raw_probe.save_checkpoint(str(out_dir / "pretrain.pt"))
        config_data = {
            "phase": "pretrain",
            "d_model": args.d_model,
            "d_hidden": args.d_hidden,
            "d_probe": args.d_probe,
            "lr": args.lr,
            "focal_gamma": args.focal_gamma,
            "batch_size": args.batch_size,
            "n_disruption_heads": len(disruption_specs),
            "total_steps": step,
        }
        (out_dir / "pretrain_config.json").write_text(json.dumps(config_data, indent=2))
        logger.info(f"Saved: {out_dir / 'pretrain.pt'} ({step} steps)")

        # Upload checkpoint + config as wandb artifact
        artifact = wandb.Artifact(f"pretrain-{args.name}", type="model")
        artifact.add_file(str(out_dir / "pretrain.pt"))
        artifact.add_file(str(out_dir / "pretrain_config.json"))
        wandb.log_artifact(artifact)


# ── Phase 2: Variant finetuning ──────────────────────────────────────────


def finetune(args, device, rank, world_size):
    """Finetune on variants: effect heads (diff), disruption heads (ref view)."""
    disruption_specs, effect_specs = load_head_specs()
    all_specs = {**disruption_specs, **effect_specs}

    head_names = list(all_specs.keys())
    disruption_indices = [i for i, name in enumerate(head_names) if name in disruption_specs]
    effect_indices = [i for i, name in enumerate(head_names) if name in effect_specs]
    specs_tuple = tuple(all_specs.values())

    # Load annotations + gene split
    variants = load_variants()
    annotations = variants.filter(pl.col("label") != "VUS")
    train_df, test_df = gene_split(annotations, test_size=args.test_size, seed=args.seed)
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

    # Freeze disruption heads only if pretrain checkpoint was loaded
    if args.checkpoint:
        disruption_prefixes = tuple(f"head_modules.{name}." for name in disruption_specs)
        for param_name, param in probe.named_parameters():
            if any(param_name.startswith(dp) for dp in disruption_prefixes):
                param.requires_grad_(False)
        logger.info(f"Froze {len(disruption_specs)} disruption heads (pretrained)")

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

    # Sync step counts across DDP ranks to prevent deadlocks from uneven chunk sizes.
    # Workaround for goodfire-ai/goodfire-core#293 (not yet merged).
    if distributed:
        steps = torch.tensor([var_iter.steps_per_epoch], device=device)
        dist.all_reduce(steps, op=dist.ReduceOp.MIN)
        var_iter.set_max_steps(int(steps.item()))

    out_dir = args.activations / "probe_v11"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        pl.concat([
            train_df.select("variant_id", "gene_name").with_columns(pl.lit("train").alias("split")),
            test_df.select("variant_id", "gene_name").with_columns(pl.lit("test").alias("split")),
        ]).write_ipc(out_dir / "split.feather")

    global_step = 0
    for epoch in range(args.epochs):
        probe.train()
        pbar = tqdm(var_iter.iter_epoch(), total=var_iter.steps_per_epoch,
                    desc=f"finetune e{epoch+1}", disable=rank != 0)

        for batch in pbar:
            raw = batch.acts
            n = raw.shape[0]

            # Look up labels
            indices = torch.tensor([id_to_idx.get(sid, -1) for sid in batch.sequence_ids], dtype=torch.long)
            known = indices >= 0
            labels = missing.unsqueeze(0).expand(n, -1).clone()
            if known.any():
                labels[known] = label_matrix[indices[known]]
            labels = labels.to(device, non_blocking=True)

            # Mask by view
            ref_labels = labels.clone()
            diff_labels = labels.clone()
            for i in effect_indices:
                ref_labels[:, i] = missing[i]
            for i in disruption_indices:
                diff_labels[:, i] = missing[i]

            # Pass 1: diff → effect heads
            diff_logits = probe(unified_diff(raw).float())
            loss_diff = multihead_loss(diff_logits, diff_labels, specs_tuple, focal_gamma=args.focal_gamma, class_balance=True)

            # Pass 2: ref → global ref heads (locals frozen)
            ref_logits = probe(unified_ref(raw).float())
            loss_ref = multihead_loss(ref_logits, ref_labels, specs_tuple, focal_gamma=args.focal_gamma, class_balance=True)

            loss = loss_diff + loss_ref
            optimizer.zero_grad()
            loss.backward()

            if rank == 0:
                raw_p = probe.module if distributed else probe
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_p.parameters(), max_norm=float("inf"))
                wandb.log({
                    "finetune/loss": loss.item(),
                    "finetune/loss_diff": loss_diff.item(),
                    "finetune/loss_ref": loss_ref.item(),
                    "finetune/grad_norm": grad_norm.item(),
                    "finetune/epoch": epoch,
                    "finetune/step": global_step,
                    "finetune/batch_size": n,
                    "finetune/known_labels_frac": known.float().mean().item(),
                })
                pbar.set_postfix(loss=f"{loss.item():.3f}", diff=f"{loss_diff.item():.3f}", ref=f"{loss_ref.item():.3f}")

            optimizer.step()
            global_step += 1

        # ── End-of-epoch checkpoint ───────────────────────────────────────
        if rank == 0:
            raw_probe = probe.module if distributed else probe
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            raw_probe.save_checkpoint(str(ckpt_path))
            logger.info(f"Checkpoint: {ckpt_path}")

    # ── Save final + upload artifact ──────────────────────────────────────
    if rank == 0:
        raw_probe = probe.module if distributed else probe
        raw_probe.save_checkpoint(str(out_dir / "weights.pt"))
        config_data = {
            "phase": "finetune",
            "preset": args.preset,
            "d_model": args.d_model,
            "d_hidden": args.d_hidden,
            "d_probe": args.d_probe,
            "lr": args.lr,
            "epochs": args.epochs,
            "focal_gamma": args.focal_gamma,
            "n_effect": len(effect_specs),
            "n_disruption": len(disruption_specs),
            "pretrain": str(args.checkpoint) if args.checkpoint else None,
            "n_train": train_df.height,
            "n_test": test_df.height,
        }
        (out_dir / "config.json").write_text(json.dumps(config_data, indent=2))
        logger.info(f"Saved: {out_dir / 'weights.pt'}")

        # Upload everything as wandb artifact
        artifact = wandb.Artifact(f"probe-{args.name}", type="model")
        artifact.add_file(str(out_dir / "weights.pt"))
        artifact.add_file(str(out_dir / "config.json"))
        if (out_dir / "eval.json").exists():
            artifact.add_file(str(out_dir / "eval.json"))
        if (out_dir / "split.feather").exists():
            artifact.add_file(str(out_dir / "split.feather"))
        wandb.log_artifact(artifact)


# ── DDP setup ─────────────────────────────────────────────────────────────


def _init_distributed() -> tuple[torch.device, int, int, bool]:
    """Initialize DDP if running under torchrun, else single GPU."""
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
    return device, rank, world_size, distributed


# ── CLI ───────────────────────────────────────────────────────────────────

app = typer.Typer(help="Probe training: reference pretrain + variant finetune")


@app.command()
def pretrain_cmd(
    name: str = typer.Option("probe-next-pretrain", help="W&B run name"),
    activations: Path = typer.Option(DECONFOUNDED, help="Activations storage directory"),
    d_model: int = typer.Option(8192),
    d_hidden: int = typer.Option(64),
    d_probe: int = typer.Option(256),
    lr: float = typer.Option(0.01),
    batch_size: int = typer.Option(2048),
    focal_gamma: float = typer.Option(3.0),
    seed: int = typer.Option(42),
) -> None:
    """Phase 1: Pretrain ref heads on single-token SAE activations."""
    torch.manual_seed(seed)
    device, rank, world_size, distributed = _init_distributed()
    if os.environ.get("SLURM_JOB_ID"):
        os.environ.setdefault("WANDB_DIR", f"/tmp/wandb_{os.environ['SLURM_JOB_ID']}")
    args = SimpleNamespace(**{k: v for k, v in locals().items() if k not in ("distributed",)})
    if rank == 0:
        wandb.init(project="gfm-probes", name=name, config=vars(args))
    pretrain(args, device, rank, world_size)
    if distributed:
        dist.destroy_process_group()


@app.command()
def finetune_cmd(
    name: str = typer.Option("probe-next-finetune", help="W&B run name"),
    activations: Path = typer.Option(DECONFOUNDED, help="Activations storage directory"),
    checkpoint: Path = typer.Option(None, help="Pretrain checkpoint (.pt)"),
    preset: str = typer.Option("deconfounded-full"),
    d_model: int = typer.Option(8192),
    d_hidden: int = typer.Option(64),
    d_probe: int = typer.Option(256),
    epochs: int = typer.Option(1),
    lr: float = typer.Option(0.01),
    batch_size: int = typer.Option(2048),
    focal_gamma: float = typer.Option(3.0),
    test_size: float = typer.Option(0.2),
    seed: int = typer.Option(42),
) -> None:
    """Phase 2: Finetune effect + global ref heads on variant activations."""
    torch.manual_seed(seed)
    device, rank, world_size, distributed = _init_distributed()
    if os.environ.get("SLURM_JOB_ID"):
        os.environ.setdefault("WANDB_DIR", f"/tmp/wandb_{os.environ['SLURM_JOB_ID']}")
    args = SimpleNamespace(**{k: v for k, v in locals().items() if k not in ("distributed",)})
    if rank == 0:
        wandb.init(project="gfm-probes", name=name, config=vars(args))
    finetune(args, device, rank, world_size)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    app()
