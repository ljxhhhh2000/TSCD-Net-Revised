"""Window pooling and classification layers for the chronological revision."""

from __future__ import annotations

import torch
from torch import nn


def build_chronological_heads(
    windows: int, symptoms: int, model_dim: int, dropout: float,
) -> tuple[nn.Embedding, nn.Sequential]:
    positions = nn.Embedding(windows, model_dim)
    distribution_head = nn.Sequential(
        nn.Linear(3 * symptoms, model_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(model_dim, 1),
    )
    return positions, distribution_head


def _symptom_statistics(
    scores: torch.Tensor, lengths: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    valid = valid.unsqueeze(1)
    centered = ((scores.float() - 0.5) * 10).masked_fill(~valid, 0.0)
    counts = lengths.unsqueeze(1).clamp_min(1)
    mean = centered.sum(dim=-1) / counts
    second_moment = centered.square().sum(dim=-1) / counts
    std = (second_moment - mean.square()).clamp_min(1e-8).sqrt()
    changes = centered.diff(dim=-1).abs().masked_fill(~valid[..., 1:], 0.0)
    mean_change = changes.sum(dim=-1) / (counts - 1).clamp_min(1)
    return torch.cat((mean, std, mean_change), dim=1)


def _pool_chronological_windows(
    scores: torch.Tensor,
    gaps: torch.Tensor,
    lengths: torch.Tensor,
    windows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Divide each valid history into consecutive, nearly equal post counts.
    # Prefix sums pool all channels without touching any of the right padding.
    counts = lengths.clamp(max=windows)
    bins = torch.arange(windows + 1, device=lengths.device).unsqueeze(0)
    boundaries = bins.minimum(counts.unsqueeze(1)) * lengths.unsqueeze(1)
    boundaries = boundaries // counts.unsqueeze(1).clamp_min(1)
    values = torch.cat((scores, gaps.unsqueeze(1)), dim=1)
    prefix = torch.nn.functional.pad(values.cumsum(dim=-1, dtype=torch.float32), (1, 0))
    prefix_at_boundaries = prefix.gather(
        -1, boundaries.unsqueeze(1).expand(-1, values.shape[1], -1)
    )
    pooled = prefix_at_boundaries.diff(dim=-1) / boundaries.diff(dim=-1).clamp_min(1).unsqueeze(1)
    return pooled[:, :-1], pooled[:, -1], counts
