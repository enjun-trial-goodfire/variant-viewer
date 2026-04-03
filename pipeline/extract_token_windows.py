"""Extract window-mean disruption scores from the token probe.

For each variant, runs the probe on ref and var views at all 512 positions.
For each head, computes mean ref, var, and signed delta (var - ref) within
genomic windows centered on the variant position:
  - w{r}_ref_{head}: mean ref score within window
  - w{r}_var_{head}: mean var score within window
  - w{r}_delta_{head}: mean signed delta within window
where r ∈ {0, 2, 64} (bp radius; w0 uses ±1 bp flanking tokens).

The variant position itself is not among the 512 scored positions (harvest
excludes it via anchor alignment). For w0, the ±1 bp flanking tokens are used.

Parallelization: use SLURM array jobs to shard across GPUs. Each shard writes
a separate feather file; finalize step merges them.

Usage:
    # Single GPU:
    uv run python pipeline/extract_token_windows.py --n-shards 1
    # 32 GPUs via SLURM array:
    sbatch --array=0-31 pipeline/extract_token_windows.sh
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

SCORE_CHUNK = 4096  # tokens per forward pass (fits ~80 GB)
WINDOW_RADII = (0, 2, 64)


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
                probs = torch.softmax(head_logits, dim=-1)
                centers = (torch.arange(size, device=probs.device).float() + 0.5) / size
                chunk_scores.append((probs * centers).sum(-1))
        all_scores.append(torch.stack(chunk_scores, dim=1))
    return torch.cat(all_scores, dim=0).reshape(B, K, n_heads)


def window_means(
    ref_probs: torch.Tensor,
    var_probs: torch.Tensor,
    all_pos: torch.Tensor,
    variant_pos: int,
    n_heads: int,
) -> dict[str, torch.Tensor]:
    """Compute mean ref, var, and delta within genomic windows.

    Args:
        ref_probs: [512, n_heads] ref scores at each position.
        var_probs: [512, n_heads] var scores at each position.
        all_pos: [512] genomic coordinates (fwd 256 + bwd 256).
        variant_pos: 0-based genomic position of the variant.
        n_heads: number of probe heads.

    Returns:
        {f"w{r}_{metric}": tensor [n_heads]} for each window radius
        and metric in (ref, var, delta).
    """
    dist = (all_pos - variant_pos).abs()
    valid = all_pos >= 0  # exclude padding (-1)
    nan = torch.full((n_heads,), float("nan"), device=ref_probs.device)

    results = {}
    for r in WINDOW_RADII:
        # w0: use ±1 bp since variant_pos itself is not in the array
        effective_r = max(r, 1)
        mask = (dist <= effective_r) & valid
        if mask.sum() == 0:
            results[f"w{r}_ref"] = nan
            results[f"w{r}_var"] = nan
            results[f"w{r}_delta"] = nan
        else:
            results[f"w{r}_ref"] = ref_probs[mask].mean(dim=0)
            results[f"w{r}_var"] = var_probs[mask].mean(dim=0)
            results[f"w{r}_delta"] = (var_probs[mask] - ref_probs[mask]).mean(dim=0)
    return results


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

    # Shard by chunks
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
    shard_path = probe_dir / f"token_window_scores_shard_{shard_id}.feather"
    if shard_path.exists() and mode == "continue":
        logger.info(f"Shard {shard_id} complete, skipping")
        return

    # Load positions
    pos_ds = local_activation_dataset(storage, "positions", batch_size=512, include_provenance=True)
    pos_lookup: dict[str, torch.Tensor] = {}
    for cid in tqdm(range(pos_ds.num_chunks), desc="Loading positions", disable=shard_id != 0):
        chunk = pos_ds.load_chunk(cid)
        for i, sid in enumerate(chunk.sequence_ids):
            pos_lookup[sid] = chunk.acts[i]

    # Load chunks directly
    acts_ds = local_activation_dataset(storage, "activations", batch_size=1, include_provenance=True)

    rows: list[dict] = []
    head_sizes = probe.head_sizes

    with torch.no_grad():
        for chunk_id in tqdm(shard_chunks, desc=f"extract_win s{shard_id}"):
            chunk = acts_ds.load_chunk(chunk_id)
            raw = chunk.acts.float().to(device)

            ref_probs = score_tokens(probe, unpack_ref(raw), head_sizes)
            var_probs = score_tokens(probe, unpack_var(raw), head_sizes)
            delta = var_probs - ref_probs  # [B, 512, n_heads]

            for i, sid in enumerate(chunk.sequence_ids):
                # Parse variant_pos from variant_id (format: chr{N}:{pos}:{ref}:{alt})
                vp = int(sid.split(":")[1])

                positions = pos_lookup.get(sid)
                row = {"variant_id": sid}

                if positions is None:
                    for r in WINDOW_RADII:
                        for metric in ("ref", "var", "delta"):
                            for name in head_names:
                                row[f"w{r}_{metric}_{name}"] = float("nan")
                else:
                    all_pos = torch.cat([positions[0], positions[1]])  # [512]
                    wm = window_means(
                        ref_probs[i], var_probs[i], all_pos.to(device), vp, n_heads,
                    )
                    for r in WINDOW_RADII:
                        for metric in ("ref", "var", "delta"):
                            vals = wm[f"w{r}_{metric}"].cpu().numpy()
                            for j, name in enumerate(head_names):
                                row[f"w{r}_{metric}_{name}"] = float(vals[j])

                rows.append(row)

    df = pl.DataFrame(rows)
    df.write_ipc(shard_path)
    logger.info(f"Shard {shard_id}: {df.height:,} variants × {df.width} cols → {shard_path}")


if __name__ == "__main__":
    typer.run(main)
