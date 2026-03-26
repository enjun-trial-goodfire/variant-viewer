"""Multi-head covariance probe v2: mixed categorical + continuous heads.

Extends v1 with:
- Soft-binned continuous heads (smooth regression via classification)
- Cross-feature heads (consequence x pathogenic)
- Amino acid swap head
- Per-domain Pfam heads

The loss function handles three head types:
- "categorical": standard cross-entropy with integer labels, -1 = masked
- "continuous": soft cross-entropy on binned targets, NaN = masked
- "binary": same as categorical with 2 classes (just for clarity)

Architecture::

    x → [shared covariance embedding] → cov [B, d_hidden, d_hidden]
         ├→ categorical heads → cross_entropy(logits, label)
         └→ continuous heads  → soft_cross_entropy(logits, soft_bins(value))

Example:
    >>> heads = {"pathogenic": Head(2, "categorical"), "phylop": Head(16, "continuous")}
    >>> probe = MultiHeadCovProbeV2(d_model=8192, heads=heads)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional
from goodfire_core.probes.covariance import newton_schulz_sqrtm
from torch import Tensor, nn

from probe.binning import create_soft_bins


@dataclass(frozen=True)
class HeadSpec:
    """Specification for a single probe head."""

    n_classes: int
    kind: str = "categorical"  # "categorical", "continuous", or "binary"
    weight: float = 1.0

    @property
    def n_outputs(self) -> int:
        return self.n_classes


class _FactoredHead(nn.Module):
    """Factored bilinear head: W_left @ cov @ W_right^T → Linear → logits."""

    def __init__(self, d_hidden: int, d_probe: int, n_outputs: int):
        super().__init__()
        self.head_left = nn.Linear(d_hidden, d_probe, bias=False)
        self.head_right = nn.Linear(d_hidden, d_probe, bias=False)
        self.out = nn.Linear(d_probe, n_outputs)

    def forward(self, cov: Tensor) -> Tensor:
        hidden = torch.einsum(
            "blr,hl,hr->bh",
            cov, self.head_left.weight, self.head_right.weight,
        )
        return self.out(hidden)


class MultiHeadCovProbeV2(nn.Module):
    """Multi-head covariance probe with mixed categorical + continuous heads.

    Args:
        d_model: Input activation dimension.
        heads: Ordered mapping of head name → HeadSpec.
        d_hidden: Covariance embedding dimension.
        d_probe: Factored probe hidden dimension per head.
        n_sqrtm_iters: Newton-Schulz iterations.
        eps: Diagonal regularization.
    """

    def __init__(
        self,
        d_model: int,
        heads: dict[str, HeadSpec],
        d_hidden: int = 64,
        d_probe: int = 128,
        n_sqrtm_iters: int = 3,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.d_model = d_model
        self.heads = OrderedDict(heads)
        self.d_hidden = d_hidden
        self.d_probe = d_probe
        self.n_sqrtm_iters = n_sqrtm_iters
        self.eps = eps

        # Shared covariance backbone
        self.proj_left = nn.Linear(d_model, d_hidden)
        self.proj_right = nn.Linear(d_model, d_hidden)
        self.register_buffer("_eye", torch.eye(d_hidden))

        # Per-head factored bilinear classifiers
        self.head_modules = nn.ModuleDict({
            name: _FactoredHead(d_hidden, d_probe, spec.n_outputs)
            for name, spec in heads.items()
        })

    @property
    def name(self) -> str:
        return f"multihead_v2_cov{self.d_hidden}"

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(self.heads.keys())

    @property
    def head_sizes(self) -> tuple[int, ...]:
        return tuple(spec.n_outputs for spec in self.heads.values())

    @property
    def n_outputs(self) -> int:
        return sum(self.head_sizes)

    def _resolve_mask(
        self, x: Tensor, attn_mask: Tensor | None, lengths: Tensor | None,
    ) -> Tensor | None:
        if attn_mask is None and lengths is not None:
            return torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        return attn_mask

    def embedding(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Shared covariance embedding [B, d_hidden, d_hidden]."""
        left = self.proj_left(x).float()
        right = self.proj_right(x).float()

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1).float()
            left = left * mask
            right = right * mask
            lengths = attn_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
            cov = torch.einsum("bsl,bsr->blr", left, right) / lengths.clamp(min=1)
        else:
            cov = torch.einsum("bsl,bsr->blr", left, right) / x.size(-2)

        cov = cov + self.eps * self._eye
        if self.n_sqrtm_iters > 0:
            cov = newton_schulz_sqrtm(cov, self.n_sqrtm_iters)
        return cov

    def forward(
        self, x: Tensor, attn_mask: Tensor | None = None, lengths: Tensor | None = None,
    ) -> Tensor:
        """Packed logits [B, sum(head_sizes)]."""
        cov = self.embedding(x, attn_mask=self._resolve_mask(x, attn_mask, lengths))
        return torch.cat([self.head_modules[n](cov) for n in self.head_names], dim=-1)

    def forward_dict(
        self, x: Tensor, attn_mask: Tensor | None = None, lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Per-head logits as a dict."""
        cov = self.embedding(x, attn_mask=self._resolve_mask(x, attn_mask, lengths))
        return {name: self.head_modules[name](cov) for name in self.head_names}

    # ── Checkpoint ────────────────────────────────────────────────────────

    def save_checkpoint(self, checkpoint_path: str, **kwargs) -> None:
        model_config = {
            "d_model": self.d_model,
            "heads": {
                name: {"n_classes": spec.n_classes, "kind": spec.kind, "weight": spec.weight}
                for name, spec in self.heads.items()
            },
            "d_hidden": self.d_hidden,
            "d_probe": self.d_probe,
            "n_sqrtm_iters": self.n_sqrtm_iters,
            "eps": self.eps,
        }
        torch.save({"model_config": model_config, "state_dict": self.state_dict(), **kwargs}, checkpoint_path)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> MultiHeadCovProbeV2:
        checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["model_config"]
        heads = {name: HeadSpec(**spec) for name, spec in config.pop("heads").items()}
        probe = cls(heads=heads, **config)
        probe.load_state_dict(checkpoint["state_dict"])
        probe.eval()
        return probe


# ── Loss function ─────────────────────────────────────────────────────────


def _soft_cross_entropy(
    logits: Tensor, targets: Tensor, n_bins: int, sigma: float = 0.05,
) -> Tensor:
    """Cross-entropy between logits and soft-binned continuous targets.

    Args:
        logits: [N, n_bins] raw logits.
        targets: [N] continuous values in [0, 1].
    """
    # create_soft_bins expects [batch, n_tracks, 1] → [batch, n_tracks, n_bins]
    soft_targets = create_soft_bins(targets.unsqueeze(-1), n_bins=n_bins, sigma=sigma)
    soft_targets = soft_targets.squeeze(1)  # [N, n_bins]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def multihead_loss_v2(
    logits: Tensor,
    labels: Tensor,
    head_specs: tuple[HeadSpec, ...],
) -> Tensor:
    """Multi-task loss with mixed categorical + continuous heads.

    Batches heads by (kind, n_classes, weight) to minimize kernel launches.
    With 258 heads this is ~10x fewer launches than the per-head loop.

    Args:
        logits: Packed logits [B, sum(head_outputs)].
        labels: [B, n_heads] where:
            - Categorical/binary heads: integer class index, -1 = masked
            - Continuous heads: float value in [0, 1], NaN = masked
        head_specs: Tuple of HeadSpec for each head.

    Returns:
        Scalar weighted loss.
    """
    # Group heads by (kind, n_classes, weight) for batched computation
    groups: dict[tuple[str, int, float], list[tuple[int, int]]] = {}
    offset = 0
    for i, spec in enumerate(head_specs):
        key = (spec.kind, spec.n_classes, spec.weight)
        groups.setdefault(key, []).append((i, offset))
        offset += spec.n_outputs

    total_loss = logits.new_tensor(0.0)

    for (kind, n_classes, weight), members in groups.items():
        head_indices = [m[0] for m in members]
        logit_offsets = [m[1] for m in members]

        # Stack logits: [B, n_heads_in_group, n_classes]
        group_logits = torch.stack(
            [logits[:, o:o + n_classes] for o in logit_offsets], dim=1,
        )
        group_labels = labels[:, head_indices]  # [B, n_heads_in_group]

        if kind == "continuous":
            # Valid mask: not NaN
            valid = ~torch.isnan(group_labels)  # [B, n_heads_in_group]
            n_valid = valid.sum()
            if n_valid == 0:
                continue
            # Soft cross-entropy on valid entries
            flat_logits = group_logits[valid]  # [N_valid, n_bins]
            flat_targets = group_labels[valid]  # [N_valid]
            soft_targets = create_soft_bins(flat_targets.unsqueeze(-1), n_bins=n_classes, sigma=0.05)
            soft_targets = soft_targets.squeeze(1)  # [N_valid, n_bins]
            log_probs = torch.nn.functional.log_softmax(flat_logits, dim=-1)
            loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        else:
            # Valid mask: label >= 0
            valid = group_labels >= 0  # [B, n_heads_in_group]
            n_valid = valid.sum()
            if n_valid == 0:
                continue
            flat_logits = group_logits[valid]  # [N_valid, n_classes]
            flat_labels = group_labels[valid].long()  # [N_valid]
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels)

        total_loss = total_loss + weight * loss

    return total_loss
