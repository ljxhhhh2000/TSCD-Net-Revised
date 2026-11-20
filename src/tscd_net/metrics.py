from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a threshold on validation data, preferring 0.5 on ties."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    total = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    ties = thresholds[scores == scores.max()]
    return float(ties[np.abs(ties - 0.5).argmin()])


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return paper Eqs. (8)-(11), treating depressed=1 as positive."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

