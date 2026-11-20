from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def stratified_70_10_20(
    labels: np.ndarray,
    seed: int,
) -> DatasetSplit:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Expected binary labels 0 and 1")

    rng = np.random.default_rng(seed)
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 10:
            raise ValueError(f"Class {label} has too few users")
        rng.shuffle(indices)
        train_count = int(round(0.7 * len(indices)))
        validation_count = int(round(0.1 * len(indices)))
        train.append(indices[:train_count])
        validation.append(
            indices[train_count : train_count + validation_count]
        )
        test.append(indices[train_count + validation_count :])

    result = DatasetSplit(
        train=np.sort(np.concatenate(train)),
        validation=np.sort(np.concatenate(validation)),
        test=np.sort(np.concatenate(test)),
    )
    _validate_split(labels, result)
    return result


def _validate_split(labels: np.ndarray, split: DatasetSplit) -> None:
    all_indices = np.concatenate([split.train, split.validation, split.test])
    if len(all_indices) != len(labels) or len(np.unique(all_indices)) != len(
        all_indices
    ):
        raise ValueError("Split indices are not a disjoint partition")
    for subset in (split.train, split.validation, split.test):
        if len(np.unique(labels[subset])) != 2:
            raise ValueError("Every split must contain both classes")
