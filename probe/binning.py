"""Soft binning utilities for converting continuous values to categorical distributions."""
import torch
import torch.nn as nn
import torch.nn.functional
from torch import Tensor


def r2_score(preds: Tensor, targets: Tensor) -> Tensor:
    """Per-task R² score.

    Args:
        preds: Predictions, shape (N,), (..., N), or (..., N, L)
        targets: True values, same shape as preds

    Returns:
        R² per task. Returns 0 for constant-target columns.
    """
    squeezed = preds.dim() == 1
    if squeezed:
        preds, targets = preds.unsqueeze(-1), targets.unsqueeze(-1)

    ss_res = ((targets - preds) ** 2).sum(-2)
    ss_tot = ((targets - targets.mean(-2, keepdim=True)) ** 2).sum(-2)
    valid = ss_tot > 1e-10
    result = torch.where(valid, 1 - ss_res / ss_tot.clamp(min=1e-10), torch.zeros_like(ss_tot))
    return result.squeeze(-1) if squeezed else result


def create_soft_bins(values: torch.Tensor, n_bins: int = 64, sigma: float = 0.1) -> torch.Tensor:
    """Convert continuous values to soft categorical distributions using Gaussian smoothing."""
    # Create bin centers at midpoints of intervals [i/n_bins, (i+1)/n_bins]
    bin_centers = (torch.arange(n_bins, device=values.device, dtype=values.dtype) + 0.5) / n_bins

    # Reshape for broadcasting: values (batch, n_tracks, 1), bin_centers (1, 1, n_bins)
    values_expanded = values.unsqueeze(-1)
    bin_centers_expanded = bin_centers.view(1, 1, -1)

    # Compute Gaussian distances
    gaussian = torch.exp(-0.5 * ((values_expanded - bin_centers_expanded) / sigma) ** 2)

    # Normalize to probability distribution (sum to 1 over bins)
    soft_labels = gaussian / gaussian.sum(dim=-1, keepdim=True)

    return soft_labels


def bins_to_continuous(logits: torch.Tensor, n_bins: int = 64) -> torch.Tensor:
    """Convert binned logits back to continuous predictions via softmax expectation."""
    batch_size = logits.shape[0]
    n_tracks = logits.shape[1] // n_bins

    # Reshape to (batch, n_tracks, n_bins)
    logits = logits.reshape(batch_size, n_tracks, n_bins)

    # Convert to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Create bin centers at midpoints of intervals [i/n_bins, (i+1)/n_bins]
    bin_centers = (torch.arange(n_bins, device=logits.device, dtype=logits.dtype) + 0.5) / n_bins

    # Weighted average: sum over bins
    predictions = (probs * bin_centers.view(1, 1, -1)).sum(dim=-1)

    return predictions


class SoftCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with soft targets for binned regression."""

    def __init__(self, n_bins: int = 64, sigma: float = 0.1):
        super().__init__()
        self.n_bins = n_bins
        self.sigma = sigma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute soft cross-entropy loss, masking NaN targets.

        NaN targets are excluded from the loss computation. When no NaNs
        are present, behavior is identical to a simple mean.
        """
        # Mask NaN targets before binning
        mask = ~torch.isnan(targets)  # (batch, n_tracks)
        targets = targets.nan_to_num(0.0)  # safe for create_soft_bins

        # Convert targets to soft bins: (batch, n_tracks, n_bins)
        soft_targets = create_soft_bins(targets, n_bins=self.n_bins, sigma=self.sigma)

        # Reshape logits: (batch, n_tracks * n_bins) -> (batch, n_tracks, n_bins)
        batch_size = logits.shape[0]
        n_tracks = logits.shape[1] // self.n_bins
        logits = logits.reshape(batch_size, n_tracks, self.n_bins)

        # Cross-entropy with soft targets, masked for NaN entries
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        per_track = -(soft_targets * log_probs).sum(dim=-1)  # (batch, n_tracks)
        loss = (per_track * mask).sum() / mask.sum().clamp(min=1)

        return loss


def binned_regression_metrics(probe, x: torch.Tensor, y: torch.Tensor, n_bins: int = 64) -> dict:
    """Compute regression metrics for binned regression."""
    logits = probe.logits(x)
    predictions = bins_to_continuous(logits, n_bins=n_bins)

    return {
        "mse": ((predictions - y) ** 2).mean().item(),
        "r2": r2_score(predictions, y).mean().item(),
    }
