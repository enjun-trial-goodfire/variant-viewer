"""Extract per-token disruption scores from the token probe.

For each variant, runs the probe on ref and var views at all 512 positions.
For each head, finds the position with the largest |ref - var| delta and stores:
  - delta_{head}: signed delta at argmax |delta| position
  - ref_{head}: ref prediction at that position
  - var_{head}: var prediction at that position
  - dist_{head}: genomic distance from variant to peak position
  - spread_{head}: positions with |delta| > 20% of peak (relative threshold)

Parallelization: use SLURM array jobs to shard across GPUs. Each shard writes
a separate feather file; finalize step merges them.

Usage:
    # Single GPU:
    uv run python pipeline/extract_token.py --n-shards 1
    # 32 GPUs via SLURM array:
    sbatch --array=0-31 --gpus-per-task=1 pipeline/extract_token.sh
"""

from pathlib import Path

import polars as pl
import torch
import typer
from goodfire_core.storage import FilesystemStorage
from loguru import logger
from tqdm import tqdm

from paths import DECONFOUNDED
from probe.covariance import MultiHeadCovProbeV2
from training import local_activation_dataset

SPREAD_FRACTION = 0.2  # positions with |delta| > 20% of peak count as "spread"
SCORE_CHUNK = 4096  # tokens per forward pass (fits ~80 GB)


def unpack_ref(raw: torch.Tensor) -> torch.Tensor:
    """[B, 2, 3, 256, 4096] → [B, 512, 1, 8192] ref view."""
    fwd = torch.cat([raw[:, 0, 1], raw[:, 0, 2]], dim=-1)
    bwd = torch.cat([raw[:, 1, 2], raw[:, 1, 1]], dim=-1)
    return torch.cat([fwd, bwd], dim=1).unsqueeze(2)


def unpack_var(raw: torch.Tensor) -> torch.Tensor:
    """[B, 2, 3, 256, 4096] → [B, 512, 1, 8192] var view."""
    fwd = torch.cat([raw[:, 0, 0], raw[:, 0, 2]], dim=-1)
    bwd = torch.cat([raw[:, 1, 2], raw[:, 1, 0]], dim=-1)
    return torch.cat([fwd, bwd], dim=1).unsqueeze(2)


def score_tokens(
    probe: MultiHeadCovProbeV2,
    tokens: torch.Tensor,
    head_sizes: tuple[int, ...],
) -> torch.Tensor:
    """[B, 512, 1, 8192] → [B, 512, n_heads] score per head.

    Binary heads (2-class): P(class=1).
    Continuous heads (N-class): soft-binned expectation in [0, 1].
    """
    B, K = tokens.shape[0], tokens.shape[1]
    n_heads = len(head_sizes)
    flat = tokens.reshape(B * K, 1, tokens.shape[-1])
    all_scores = []
    for start in range(0, flat.shape[0], SCORE_CHUNK):
        logits = probe(flat[start : start + SCORE_CHUNK])
        chunk_scores = []
        offset = 0
        for size in head_sizes:
            head_logits = logits[:, offset : offset + size]
            offset += size
            if size == 2:
                chunk_scores.append(torch.softmax(head_logits, dim=-1)[:, 1])
            else:
                # Soft-binned expectation: sum(prob * bin_center)
                probs = torch.softmax(head_logits, dim=-1)
                centers = (torch.arange(size, device=probs.device).float() + 0.5) / size
                chunk_scores.append((probs * centers).sum(-1))
        all_scores.append(torch.stack(chunk_scores, dim=1))
    return torch.cat(all_scores, dim=0).reshape(B, K, n_heads)


def main(
    activations: Path = typer.Option(DECONFOUNDED),
    probe_name: str = typer.Option("probe_token_v1"),
    batch_size: int = typer.Option(8, help="Variants per batch"),
    shard_id: int = typer.Option(0, help="Shard index (SLURM_ARRAY_TASK_ID)"),
    n_shards: int = typer.Option(1, help="Total shards"),
    mode: str = typer.Option("continue", help="continue or overwrite"),
) -> None:
    device = torch.device("cuda")
    storage = FilesystemStorage(activations)
    probe_dir = activations / probe_name

    # Load probe
    probe = MultiHeadCovProbeV2.from_checkpoint(str(probe_dir / "weights.pt")).to(device).eval()
    head_names = list(probe.heads.keys())
    n_heads = len(head_names)
    logger.info(f"Probe: {n_heads} heads from {probe_name}")

    # Shard by chunks (not by sequence_ids) — no boundary issues, no missed variants.
    acts_meta = local_activation_dataset(storage, "activations", batch_size=1, include_provenance=True)
    n_chunks = acts_meta.num_chunks
    effective_shards = min(n_shards, n_chunks)
    if shard_id >= effective_shards:
        logger.info(f"Shard {shard_id} >= {effective_shards} chunks, nothing to do")
        return
    chunk_start = shard_id * n_chunks // effective_shards
    chunk_end = (shard_id + 1) * n_chunks // effective_shards
    shard_chunks = list(range(chunk_start, chunk_end))
    logger.info(
        f"Shard {shard_id}/{effective_shards}: chunks {chunk_start}-{chunk_end - 1} ({len(shard_chunks)} chunks)"
    )

    # Output
    shard_path = probe_dir / f"token_scores_shard_{shard_id}.feather"
    if shard_path.exists() and mode == "continue":
        logger.info(f"Shard {shard_id} complete, skipping")
        return

    # Load positions for distance computation
    pos_ds = local_activation_dataset(storage, "positions", batch_size=512, include_provenance=True)
    pos_lookup: dict[str, torch.Tensor] = {}
    for cid in tqdm(range(pos_ds.num_chunks), desc="Loading positions", disable=shard_id != 0):
        chunk = pos_ds.load_chunk(cid)
        for i, sid in enumerate(chunk.sequence_ids):
            pos_lookup[sid] = chunk.acts[i]

    # Variant genomic positions
    var_pos = dict(
        zip(
            *pl.read_parquet("data/variants.parquet", columns=["variant_id", "pos"])
            .select("variant_id", "pos")
            .to_dict(as_series=False)
            .values()
        )
    )

    # Load chunks directly — no iterator filtering, exact coverage.
    acts_ds = local_activation_dataset(storage, "activations", batch_size=1, include_provenance=True)

    rows: list[dict] = []
    arange = torch.arange(n_heads)
    head_sizes = probe.head_sizes

    with torch.no_grad():
        for chunk_id in tqdm(shard_chunks, desc=f"extract s{shard_id}"):
            chunk = acts_ds.load_chunk(chunk_id)
            raw = chunk.acts.float().to(device)

            ref_probs = score_tokens(probe, unpack_ref(raw), head_sizes)
            var_probs = score_tokens(probe, unpack_var(raw), head_sizes)
            delta = var_probs - ref_probs  # [B, 512, n_heads]
            abs_delta = delta.abs()

            for i, sid in enumerate(chunk.sequence_ids):
                d = delta[i]  # [512, n_heads]
                ad = abs_delta[i]

                # Peak position per head
                peak_idx = ad.argmax(dim=0)  # [n_heads]
                peak_delta = d[peak_idx, arange].cpu().numpy()
                peak_ref = ref_probs[i, peak_idx, arange].cpu().numpy()
                peak_var = var_probs[i, peak_idx, arange].cpu().numpy()

                # Spread: positions with |delta| > 20% of peak
                peak_abs = ad[peak_idx, arange].unsqueeze(0)  # [1, n_heads]
                spread = (ad > SPREAD_FRACTION * peak_abs).sum(dim=0).cpu().numpy()

                # Distance at peak
                positions = pos_lookup.get(sid)
                vp = var_pos.get(sid, 0)
                if positions is not None:
                    all_pos = torch.cat([positions[0], positions[1]]).numpy()
                    peak_dist = (all_pos - vp)[peak_idx.cpu().numpy()]
                else:
                    peak_dist = [0] * n_heads

                row = {"variant_id": sid}
                for j, name in enumerate(head_names):
                    row[f"delta_{name}"] = float(peak_delta[j])
                    row[f"ref_{name}"] = float(peak_ref[j])
                    row[f"var_{name}"] = float(peak_var[j])
                    row[f"dist_{name}"] = int(peak_dist[j])
                    row[f"spread_{name}"] = int(spread[j])
                rows.append(row)

    df = pl.DataFrame(rows)
    df.write_ipc(shard_path)
    logger.info(f"Shard {shard_id}: {df.height:,} variants × {df.width} cols → {shard_path}")


if __name__ == "__main__":
    typer.run(main)
