from __future__ import annotations

import numpy as np

from ..config import ModelConfig, TrainingConfig
from .torch_trainer import TSCDClassifier


SUPPORTED_MODELS = {"tscdnet_chronological"}


def build_classifier(
    name: str,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    seed: int,
    symptom_embeddings: np.ndarray,
) -> TSCDClassifier:
    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown classifier {name!r}; supported values are "
            f"{sorted(SUPPORTED_MODELS)}"
        )
    local_config = ModelConfig(**vars(model_config))
    local_config.name = name
    return TSCDClassifier(
        local_config,
        training_config,
        seed=seed,
        symptom_embeddings=symptom_embeddings,
    )
