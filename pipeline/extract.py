"""Extract 3-view embeddings + scores from a trained probe in a single pass.

Streams raw activations once, computes all three views (diff/ref/var),
and produces both covariance embeddings and per-head scores.

Two head types:
  - **disruption** heads: scored on ref and var views separately.
    Shows what changed between reference and variant allele.
  - **effect** heads: scored on diff (var-ref) view.
    Predicts variant-level properties (clinical predictors, consequence, etc.)

Outputs (written to {activations}/{probe_name}/):
  embeddings/     [3, d_hidden, d_hidden] covariance matrices per variant
                    index 0: diff (var - ref), 1: ref, 2: var
  scores.feather  ref_score_*, var_score_*, score_*, pred_* columns

Usage:
    python scripts/extract.py --probe $ACTS/probe_v9 --activations $ACTS

Parallel (SLURM array):
    EXTRACT=$(sbatch --parsable --array=0-7 scripts/extract.sh \\
        --probe $ACTS/probe_v9 --activations $ACTS)
    sbatch --dependency=afterok:${EXTRACT} scripts/finalize_embed.sh $ACTS/probe_v9
"""

import json
from pathlib import Path

import polars as pl
import torch
import typer
from goodfire_core.storage import ActivationDataset, ActivationWriter, FilesystemStorage
from loguru import logger
from tqdm import tqdm

from probe.covariance import MultiHeadCovProbeV2
from collections.abc import Callable, Iterator

# ── Activation view transforms (inlined from streaming.py) ────────────
# Layout: [B, direction=2, view=3, K, d]. direction: 0=fwd, 1=bwd. view: 0=var, 1=ref, 2=ref_cross

def unified_diff(x: torch.Tensor) -> torch.Tensor:
    diff = x[:, :, 0] - x[:, :, 1]
    return torch.cat([diff[:, 0], diff[:, 1]], dim=-1)

def unified_ref(x: torch.Tensor) -> torch.Tensor:
    ref = x[:, :, 1]
    return torch.cat([ref[:, 0], ref[:, 1]], dim=-1)

def unified_var(x: torch.Tensor) -> torch.Tensor:
    var = x[:, :, 0]
    return torch.cat([var[:, 0], var[:, 1]], dim=-1)

def iter_dataset(
    storage: FilesystemStorage, dataset_name: str, target_ids: set[str],
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *, batch_size: int = 512, dtype: torch.dtype = torch.bfloat16, device: str = "cuda",
) -> Iterator[tuple[torch.Tensor, list[str]]]:
    ds = ActivationDataset(storage, dataset_name, batch_size=batch_size, include_provenance=True)
    for batch in ds.training_iterator(device=device, n_epochs=1, shuffle=False, drop_last=False, sequence_ids=list(target_ids)):
        x = batch.acts.to(dtype=dtype)
        if transform is not None:
            x = transform(x)
        yield x, batch.sequence_ids


def _scores_from_logits(
    logits_dict: dict[str, torch.Tensor],
    probe: MultiHeadCovProbeV2,
    prefix: str = "",
    class_labels: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list]:
    """Extract per-head scores from logits with optional column prefix.

    Args:
        class_labels: Optional {head_name: (label_0, label_1, ...)} for multi-class
            heads. If provided, writes the string label instead of the integer argmax.
    """
    result: dict[str, list] = {}
    for name, logits in logits_dict.items():
        head = probe.heads[name]
        if head.kind == "continuous":
            probs = torch.softmax(logits, dim=-1).cpu()
            centers = (torch.arange(probs.size(-1), dtype=probs.dtype) + 0.5) / probs.size(-1)
            result.setdefault(f"{prefix}score_{name}", []).extend((probs * centers).sum(-1).tolist())
        elif logits.size(-1) == 2:
            result.setdefault(f"{prefix}score_{name}", []).extend(
                torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            )
        else:
            indices = logits.argmax(-1).cpu().tolist()
            labels = class_labels.get(name) if class_labels else None
            if labels:
                result.setdefault(f"{prefix}pred_{name}", []).extend(
                    labels[i] if i < len(labels) else "unknown" for i in indices
                )
            else:
                result.setdefault(f"{prefix}pred_{name}", []).extend(indices)
    return result


def main(
    probe: Path = typer.Option(..., help="Probe directory (weights.pt + config.json)"),
    activations: Path = typer.Option(..., help="Base storage with raw activations"),
    batch_size: int = typer.Option(512, help="Batch size for streaming"),
    mode: str = typer.Option("continue", help="continue or overwrite"),
    shard_id: int = typer.Option(0, help="Shard index (for parallel extraction)"),
    n_shards: int = typer.Option(1, help="Total number of shards"),
) -> None:
    # ── Load probe + validate config ──────────────────────────────────────
    model = MultiHeadCovProbeV2.from_checkpoint(str(probe / "weights.pt")).cuda().eval()
    config = json.loads((probe / "config.json").read_text())

    # Support both old naming (ref_heads/diff_heads) and new (disruption_heads/effect_heads)
    disruption_heads = frozenset(config.get("disruption_heads", config.get("ref_heads", ())))
    effect_heads = frozenset(config.get("effect_heads", config.get("diff_heads", ())))
    assert disruption_heads, "config.json missing disruption_heads (or ref_heads) — check training script"
    assert effect_heads, "config.json missing effect_heads (or diff_heads) — check training script"

    d_hidden = model.d_hidden
    logger.info(f"Probe: {len(disruption_heads)} disruption + {len(effect_heads)} effect heads, d_hidden={d_hidden}")

    # Build class label lookup for multi-class heads (consequence, aa_swap)
    # so scores.feather writes human-readable strings instead of integer argmax.
    class_labels: dict[str, tuple[str, ...]] = {}
    for name, spec in model.heads.items():
        if spec.kind == "categorical" and spec.n_classes > 2:
            labels_key = f"{name}_classes"
            if labels_key in config:
                class_labels[name] = tuple(config[labels_key])
    if class_labels:
        logger.info(f"Class labels for {len(class_labels)} multi-class heads: {list(class_labels.keys())}")

    # ── Resolve IDs + shard ───────────────────────────────────────────────
    storage = FilesystemStorage(activations)
    all_ids = ActivationDataset(storage, "activations", batch_size=1, include_provenance=True).list_sequence_ids()

    n = len(all_ids)
    effective_shards = min(n_shards, n)
    if shard_id >= effective_shards:
        logger.info(f"Shard {shard_id} >= n_shards {effective_shards}, nothing to do")
        return
    start = shard_id * n // effective_shards
    end = (shard_id + 1) * n // effective_shards
    target_ids = set(all_ids[start:end])
    logger.info(f"Shard {shard_id}/{effective_shards}: {len(target_ids):,} variants")

    # ── Output paths ──────────────────────────────────────────────────────
    output_dir = activations / probe.name
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"scores_shard_{shard_id}.feather"
    scores_path = output_dir / "scores.feather"

    exists = shard_path.exists() if n_shards > 1 else scores_path.exists()
    if exists and mode == "continue":
        logger.info(f"Shard {shard_id} complete, skipping")
        return

    # ── Embedding writer ──────────────────────────────────────────────────
    # Store [3, d_h, d_h] flattened to [3*d_h*d_h] (goodfire-core requires 2D tensors)
    output_storage = FilesystemStorage(output_dir)
    partition_id = shard_id if n_shards > 1 else None
    writer = ActivationWriter(
        output_storage, "embeddings",
        d_model=3 * d_hidden * d_hidden,
        mode="tensor",
        dtype="bfloat16",
        shuffle=False,
        shuffle_buffer_size=1024,
        partition_id=partition_id,
    )

    # ── Stream + extract ──────────────────────────────────────────────────
    ids_out: list[str] = []
    effect_scores: dict[str, list[float]] = {}
    ref_scores: dict[str, list[float]] = {}
    var_scores: dict[str, list[float]] = {}

    with torch.no_grad():
        for raw, ids in tqdm(
            iter_dataset(storage, "activations", target_ids, None,
                         batch_size=batch_size, dtype=torch.bfloat16, device="cuda"),
            desc=f"extract s{shard_id}",
        ):
            # Three views from the same raw batch
            diff_acts = unified_diff(raw).float()
            ref_acts = unified_ref(raw).float()
            var_acts = unified_var(raw).float()

            # Embeddings: stack all 3 views → [B, 3, d_h, d_h]
            emb = torch.stack([
                model.embedding(diff_acts),
                model.embedding(ref_acts),
                model.embedding(var_acts),
            ], dim=1)
            writer.add(acts=emb.flatten(1).cpu().to(torch.bfloat16), sequence_ids=ids)

            # Scores: 3 forward passes, split by head type
            diff_logits = model.forward_dict(diff_acts)
            ref_logits = model.forward_dict(ref_acts)
            var_logits = model.forward_dict(var_acts)

            # Effect heads → score_* from diff view
            for k, v in _scores_from_logits(
                {h: diff_logits[h] for h in effect_heads}, model,
                class_labels=class_labels,
            ).items():
                effect_scores.setdefault(k, []).extend(v)

            # Disruption heads → ref_score_* from ref view, var_score_* from var view
            for k, v in _scores_from_logits(
                {h: ref_logits[h] for h in disruption_heads}, model, "ref_",
                class_labels=class_labels,
            ).items():
                ref_scores.setdefault(k, []).extend(v)

            for k, v in _scores_from_logits(
                {h: var_logits[h] for h in disruption_heads}, model, "var_",
                class_labels=class_labels,
            ).items():
                var_scores.setdefault(k, []).extend(v)

            ids_out.extend(ids)

    writer.finalize()

    # ── Save scores ───────────────────────────────────────────────────────
    df = pl.DataFrame({"variant_id": ids_out, **effect_scores, **ref_scores, **var_scores})
    out_path = shard_path if n_shards > 1 else scores_path
    df.write_ipc(out_path)
    logger.info(f"Shard {shard_id}: {df.height:,} variants, {len(df.columns)} cols → {output_dir}")


if __name__ == "__main__":
    typer.run(main)
