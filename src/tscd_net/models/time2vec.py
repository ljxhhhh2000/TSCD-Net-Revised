from __future__ import annotations

import math

import torch
from torch import nn


class Time2Vec(nn.Module):
    """Time2Vec with one linear term and periodic components.

    Input gaps are already transformed as log(1 + days) by the feature
    extractor. The paper specifies d_t=16; it does not specify the internal
    split between linear and periodic terms.
    """

    def __init__(self, output_dim: int = 16) -> None:
        super().__init__()
        if output_dim < 2:
            raise ValueError("Time2Vec output_dim must be at least 2")
        periodic_dim = output_dim - 1
        self.linear_weight = nn.Parameter(torch.empty(1))
        self.linear_bias = nn.Parameter(torch.empty(1))
        self.periodic_weight = nn.Parameter(torch.empty(periodic_dim))
        self.periodic_bias = nn.Parameter(torch.empty(periodic_dim))
        nn.init.uniform_(self.linear_weight, -1.0 / math.sqrt(1.0), 1.0)
        nn.init.zeros_(self.linear_bias)
        nn.init.uniform_(
            self.periodic_weight,
            -1.0 / math.sqrt(1.0),
            1.0,
        )
        nn.init.zeros_(self.periodic_bias)

    def forward(self, gaps: torch.Tensor) -> torch.Tensor:
        if gaps.ndim == 2:
            gaps = gaps.unsqueeze(-1)
        if gaps.shape[-1] != 1:
            raise ValueError("Time2Vec expects one scalar gap per timestep")
        linear = gaps * self.linear_weight + self.linear_bias
        periodic = torch.sin(
            gaps * self.periodic_weight + self.periodic_bias
        )
        return torch.cat([linear, periodic], dim=-1)
