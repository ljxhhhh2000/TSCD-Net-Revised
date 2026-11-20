from __future__ import annotations

import torch
from torch import nn

from .chronological import (
    _pool_chronological_windows,
    _symptom_statistics,
    build_chronological_heads,
)
from .time2vec import Time2Vec


class CrossAttentionStage(nn.Module):
    def __init__(
        self,
        model_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        ffn_multiplier: int,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        values: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.norm(self.encoder(values, src_key_padding_mask=key_padding_mask))


class TSCDNet(nn.Module):
    """Temporal Symptom-aware Clinical Depression Network.

    Each step contains K symptom tokens. Cross-symptom attention acts within
    each time step; cross-temporal attention acts over the pooled step sequence.
    Padding is excluded from temporal attention and user pooling.
    """

    def __init__(
        self,
        symptom_embeddings: torch.Tensor,
        model_dim: int = 64,
        time_dim: int = 16,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        ffn_multiplier: int = 4,
        symptom_attention: bool = True,
        temporal_attention: bool = True,
        temporal_encoding: bool = True,
        chronological_windows: int | None = None,
    ) -> None:
        super().__init__()
        if symptom_embeddings.ndim != 2:
            raise ValueError("symptom_embeddings must be [symptoms, dim]")
        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        self.symptom_anchor_projection = nn.Linear(
            symptom_embeddings.shape[1],
            model_dim,
            bias=False,
        )
        self.register_buffer(
            "frozen_symptom_embeddings",
            symptom_embeddings.float().clone(),
        )
        self.token_bias = nn.Parameter(torch.zeros(model_dim))
        self.chronological_windows = chronological_windows
        self.position_embedding, self.distribution_head = (
            build_chronological_heads(
                chronological_windows, symptom_embeddings.shape[0], model_dim, dropout
            )
            if chronological_windows is not None
            else (None, None)
        )
        self.time2vec = Time2Vec(time_dim) if temporal_encoding else None
        self.time_projection = (
            nn.Linear(time_dim, model_dim, bias=False)
            if temporal_encoding
            else None
        )
        self.symptom_stage = (
            CrossAttentionStage(
                model_dim,
                heads,
                layers,
                dropout,
                ffn_multiplier,
            )
            if symptom_attention
            else None
        )
        self.temporal_stage = (
            CrossAttentionStage(
                model_dim,
                heads,
                layers,
                dropout,
                ffn_multiplier,
            )
            if temporal_attention
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(model_dim, 1)
        self.apply(_xavier_linear_weights)
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        symptom_scores: torch.Tensor,
        log_time_gaps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if symptom_scores.ndim != 3:
            raise ValueError("symptom_scores must be [batch, symptoms, time]")
        distribution_logits = None
        if self.chronological_windows is not None:
            valid = ~_padding_mask(lengths, symptom_scores.shape[-1])
            distribution_logits = self.distribution_head(
                _symptom_statistics(symptom_scores, lengths, valid)
            )
            symptom_scores, log_time_gaps, lengths = _pool_chronological_windows(
                symptom_scores, log_time_gaps, lengths, self.chronological_windows
            )
        scores = symptom_scores.transpose(1, 2)
        batch_size, time, symptoms = scores.shape
        anchors = self.symptom_anchor_projection(
            self.frozen_symptom_embeddings
        )
        tokens = (
            scores.unsqueeze(-1) * anchors.unsqueeze(0).unsqueeze(0)
            + self.token_bias
        )
        if self.time2vec is not None and self.time_projection is not None:
            temporal = self.time_projection(self.time2vec(log_time_gaps))
            tokens = tokens + temporal.unsqueeze(2)

        if self.symptom_stage is not None:
            tokens = self.symptom_stage(
                tokens.reshape(batch_size * time, symptoms, -1)
            ).reshape(batch_size, time, symptoms, -1)
        step_representation = tokens.mean(dim=2)
        if self.position_embedding is not None:
            positions = torch.arange(time, device=scores.device)
            step_representation = step_representation + self.position_embedding(positions)

        padding_mask = _padding_mask(lengths, time)
        if self.temporal_stage is not None:
            step_representation = self.temporal_stage(
                step_representation,
                key_padding_mask=padding_mask,
            )
        observed = (~padding_mask).unsqueeze(-1).to(step_representation.dtype)
        user_representation = (
            step_representation * observed
        ).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.dropout(user_representation))
        if distribution_logits is not None:
            logits = logits + distribution_logits
        return logits


def _padding_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    positions = torch.arange(width, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def _xavier_linear_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
