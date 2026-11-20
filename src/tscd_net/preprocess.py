from __future__ import annotations

import numpy as np

from .features import TSCDFeatures


def crop_series(
    features: TSCDFeatures,
    sequence_length: int,
) -> TSCDFeatures:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if sequence_length > features.symptom_scores.shape[2]:
        raise ValueError("Cannot crop to a wider sequence")
    result = TSCDFeatures(
        symptom_scores=features.symptom_scores[:, :, :sequence_length],
        log_time_gaps=features.log_time_gaps[:, :sequence_length],
        lengths=np.minimum(features.lengths, sequence_length).astype(
            np.int32,
            copy=False,
        ),
        labels=features.labels,
        symptom_truth=features.symptom_truth,
        user_ids=features.user_ids,
        symptom_names=list(features.symptom_names),
        symptom_embeddings=features.symptom_embeddings,
        embedding_model=features.embedding_model,
    )
    result.validate()
    return result
