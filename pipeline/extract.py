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
    python scripts/extract.py --probe $ACTS/probe_v8 --activations $ACTS

Parallel (SLURM array):
    EXTRACT=$(sbatch --parsable --array=0-7 scripts/extract.sh \\
        --probe $ACTS/probe_v8 --activations $ACTS)
    sbatch --dependency=afterok:${EXTRACT} scripts/finalize_embed.sh $ACTS/probe_v8
"""

import argparse
import json
from pathlib import Path

import polars as pl
import torch
from goodfire_core.storage import ActivationDataset, ActivationWriter, FilesystemStorage
from loguru import logger
from tqdm import tqdm

from probe.multihead_v2 import MultiHeadCovProbeV2
from streaming import iter_dataset, unified_diff, unified_ref, unified_var


def _scores_from_logits(
    logits_dict: dict[str, torch.Tensor],
    probe: MultiHeadCovProbeV2,
    prefix: str = "",
) -> dict[str, list[float]]:
    """Extract per-head scores from logits with optional column prefix."""
    result: dict[str, list[float]] = {}
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
            result.setdefault(f"{prefix}pred_{name}", []).extend(logits.argmax(-1).cpu().tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, required=True, help="Probe directory (weights.pt + config.json)")
    parser.add_argument("--activations", type=Path, required=True, help="Base storage with raw activations")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mode", choices=("continue", "overwrite"), default="continue")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()

    # ── Load probe + validate config ──────────────────────────────────────
    probe = MultiHeadCovProbeV2.from_checkpoint(str(args.probe / "weights.pt")).cuda().eval()
    config = json.loads((args.probe / "config.json").read_text())

    # Support both old naming (ref_heads/diff_heads) and new (disruption_heads/effect_heads)
    disruption_heads = frozenset(config.get("disruption_heads", config.get("ref_heads", ())))
    effect_heads = frozenset(config.get("effect_heads", config.get("diff_heads", ())))
    assert disruption_heads, "config.json missing disruption_heads (or ref_heads) — check training script"
    assert effect_heads, "config.json missing effect_heads (or diff_heads) — check training script"

    d_hidden = probe.d_hidden
    logger.info(f"Probe: {len(disruption_heads)} disruption + {len(effect_heads)} effect heads, d_hidden={d_hidden}")

    # ── Resolve IDs + shard ───────────────────────────────────────────────
    storage = FilesystemStorage(args.activations)
    all_ids = ActivationDataset(storage, "activations", batch_size=1, include_provenance=True).list_sequence_ids()

    n = len(all_ids)
    n_shards = min(args.n_shards, n)
    if args.shard_id >= n_shards:
        logger.info(f"Shard {args.shard_id} >= n_shards {n_shards}, nothing to do")
        return
    start = args.shard_id * n // n_shards
    end = (args.shard_id + 1) * n // n_shards
    target_ids = set(all_ids[start:end])
    logger.info(f"Shard {args.shard_id}/{n_shards}: {len(target_ids):,} variants")

    # ── Output paths ──────────────────────────────────────────────────────
    output_dir = args.activations / args.probe.name
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"scores_shard_{args.shard_id}.feather"
    scores_path = output_dir / "scores.feather"

    exists = shard_path.exists() if args.n_shards > 1 else scores_path.exists()
    if exists and args.mode == "continue":
        logger.info(f"Shard {args.shard_id} complete, skipping")
        return

    # ── Embedding writer ──────────────────────────────────────────────────
    # Store [3, d_h, d_h] flattened to [3*d_h*d_h] (goodfire-core requires 2D tensors)
    output_storage = FilesystemStorage(output_dir)
    partition_id = args.shard_id if args.n_shards > 1 else None
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
                         batch_size=args.batch_size, dtype=torch.bfloat16, device="cuda"),
            desc=f"extract s{args.shard_id}",
        ):
            # Three views from the same raw batch
            diff_acts = unified_diff(raw).float()
            ref_acts = unified_ref(raw).float()
            var_acts = unified_var(raw).float()

            # Embeddings: stack all 3 views → [B, 3, d_h, d_h]
            emb = torch.stack([
                probe.embedding(diff_acts),
                probe.embedding(ref_acts),
                probe.embedding(var_acts),
            ], dim=1)
            writer.add(acts=emb.flatten(1).cpu().to(torch.bfloat16), sequence_ids=ids)

            # Scores: 3 forward passes, split by head type
            diff_logits = probe.forward_dict(diff_acts)
            ref_logits = probe.forward_dict(ref_acts)
            var_logits = probe.forward_dict(var_acts)

            # Effect heads → score_* from diff view
            for k, v in _scores_from_logits(
                {h: diff_logits[h] for h in effect_heads}, probe,
            ).items():
                effect_scores.setdefault(k, []).extend(v)

            # Disruption heads → ref_score_* from ref view, var_score_* from var view
            for k, v in _scores_from_logits(
                {h: ref_logits[h] for h in disruption_heads}, probe, "ref_",
            ).items():
                ref_scores.setdefault(k, []).extend(v)

            for k, v in _scores_from_logits(
                {h: var_logits[h] for h in disruption_heads}, probe, "var_",
            ).items():
                var_scores.setdefault(k, []).extend(v)

            ids_out.extend(ids)

    writer.finalize()

    # ── Save scores ───────────────────────────────────────────────────────
    df = pl.DataFrame({"variant_id": ids_out, **effect_scores, **ref_scores, **var_scores})
    out_path = shard_path if args.n_shards > 1 else scores_path
    df.write_ipc(out_path)
    logger.info(f"Shard {args.shard_id}: {df.height:,} variants, {len(df.columns)} cols → {output_dir}")


if __name__ == "__main__":
    main()
