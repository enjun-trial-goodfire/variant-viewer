"""Multi-head covariance probe v8: attribution-aligned training.

Extends v2 with:
- Focal loss for binary heads (handles class imbalance)
- Attribution decoder: predicts pathogenicity from other bio head scores
  via soft top-k, forcing the backbone to learn interpretable representations
- logits_to_scores: differentiable score extraction (shared with scoring pipeline)

Architecture::

    x → [shared covariance embedding] → cov
         ├→ all heads → per-head losses (focal CE for binary, soft CE for continuous)
         └→ bio head scores → soft-top-k → attribution decoder → alignment loss
"""

from __future__ import annotations

import torch
import torch.nn.functional
from torch import Tensor, nn

from probe.binning import create_soft_bins
from probe.multihead_v2 import HeadSpec, MultiHeadCovProbeV2

# ── Utilities ──────────────────────────────────────────────────────────────


def logits_to_scores(logits: Tensor, kind: str, n_classes: int) -> Tensor:
    """Convert head logits to differentiable scalar scores [B].

    Same math as score.py:_scores_from_logits, but returns tensors for backprop.
    """
    if kind == "continuous":
        probs = torch.softmax(logits, dim=-1)
        centers = (torch.arange(n_classes, device=logits.device, dtype=logits.dtype) + 0.5) / n_classes
        return (probs * centers).sum(dim=-1)
    if kind == "binary":
        return torch.softmax(logits, dim=-1)[:, 1]
    # categorical: no scalar score
    return logits.new_full((logits.size(0),), float("nan"))


def focal_cross_entropy(logits: Tensor, labels: Tensor, gamma: float) -> Tensor:
    """Focal loss: -(1-p_t)^gamma log(p_t). Reduces to CE when gamma=0."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    targets = torch.nn.functional.one_hot(labels, logits.size(-1)).float()
    p_t = (log_probs.exp() * targets).sum(dim=-1)
    return -((1 - p_t) ** gamma * (targets * log_probs).sum(dim=-1)).mean()


# ── Loss function ──────────────────────────────────────────────────────────


def multihead_loss_v8(
    logits: Tensor,
    labels: Tensor,
    head_specs: tuple[HeadSpec, ...],
    focal_gamma: float = 3.0,
) -> Tensor:
    """Multi-task loss with focal CE for binary heads.

    Drop-in replacement for multihead_loss_v2. Identical except binary heads
    use focal loss when focal_gamma > 0.
    """
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

        group_logits = torch.stack(
            [logits[:, o:o + n_classes] for o in logit_offsets], dim=1,
        )
        group_labels = labels[:, head_indices]

        if kind == "continuous":
            valid = ~torch.isnan(group_labels)
            if valid.sum() == 0:
                continue
            flat_logits = group_logits[valid]
            flat_targets = group_labels[valid]
            soft_targets = create_soft_bins(flat_targets.unsqueeze(-1), n_bins=n_classes, sigma=0.05)
            soft_targets = soft_targets.squeeze(1)
            log_probs = torch.nn.functional.log_softmax(flat_logits, dim=-1)
            loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        else:
            valid = group_labels >= 0
            if valid.sum() == 0:
                continue
            flat_logits = group_logits[valid]
            flat_labels = group_labels[valid].long()
            if kind == "binary" and focal_gamma > 0:
                loss = focal_cross_entropy(flat_logits, flat_labels, focal_gamma)
            else:
                loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels)

        total_loss = total_loss + weight * loss

    return total_loss


# ── Probe ──────────────────────────────────────────────────────────────────


class MultiHeadCovProbeV8(MultiHeadCovProbeV2):
    """Attribution-aligned multihead covariance probe.

    Extends V2 with an optional attribution decoder that predicts pathogenicity
    from the scores of other biological heads via soft top-k selection.

    Args:
        bio_heads: Tuple of (head_name, source) where source is "diff" or "delta".
            When provided, creates learnable attribution weights alpha and bias.
            "diff" heads use the diff-view score directly.
            "delta" heads use var_score - ref_score.
    """

    def __init__(
        self,
        d_model: int,
        heads: dict[str, HeadSpec],
        d_hidden: int = 32,
        d_probe: int = 128,
        n_sqrtm_iters: int = 0,
        eps: float = 1e-3,
        bio_heads: tuple[tuple[str, str], ...] | None = None,
    ):
        super().__init__(
            d_model=d_model, heads=heads,
            d_hidden=d_hidden, d_probe=d_probe,
            n_sqrtm_iters=n_sqrtm_iters, eps=eps,
        )
        self.bio_heads = bio_heads
        if bio_heads is not None:
            self.attribution_alpha = nn.Parameter(torch.zeros(len(bio_heads)))
            self.attribution_bias = nn.Parameter(torch.zeros(1))

    def compute_bio_scores(
        self,
        diff_logits: dict[str, Tensor],
        ref_logits: dict[str, Tensor],
        var_logits: dict[str, Tensor],
    ) -> Tensor:
        """Compute bio feature scores [B, n_bio] for the attribution decoder.

        For "diff" heads: scalar score from diff-view logits.
        For "delta" heads: var_score - ref_score (disruption signal).
        """
        scores = []
        for name, source in self.bio_heads:
            spec = self.heads[name]
            if source == "diff":
                scores.append(logits_to_scores(diff_logits[name], spec.kind, spec.n_classes))
            else:
                var_score = logits_to_scores(var_logits[name], spec.kind, spec.n_classes)
                ref_score = logits_to_scores(ref_logits[name], spec.kind, spec.n_classes)
                scores.append(var_score - ref_score)
        return torch.stack(scores, dim=-1)

    def attribution_forward(self, bio_scores: Tensor, tau: float = 1.0) -> Tensor:
        """Predict pathogenicity logit from bio scores via soft top-k.

        Args:
            bio_scores: [B, n_bio] scores from compute_bio_scores.
            tau: Temperature for soft top-k. High → uniform, low → hard top-k.

        Returns:
            [B] pathogenicity logits.
        """
        weights = torch.softmax(self.attribution_alpha.abs() * bio_scores.abs() / tau, dim=-1)
        return (weights * self.attribution_alpha * bio_scores).sum(dim=-1) + self.attribution_bias

    # ── Checkpoint ─────────────────────────────────────────────────────────

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
            "bio_heads": self.bio_heads,
        }
        torch.save({"model_config": model_config, "state_dict": self.state_dict(), **kwargs}, checkpoint_path)
