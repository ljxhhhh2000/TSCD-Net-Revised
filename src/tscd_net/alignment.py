from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import Config
from .features import check_feature_config, load_features, select_symptom_channels
from .preprocess import crop_series


def evaluate_user_symptom_alignment(config: Config) -> Path:
    """Evaluate coarse symptom ranking with released user-level indicators.

    The local SWDD release exposes 15 symptom indicators under
    `label.symptoms`, not on each post. This report therefore must not be
    labelled as the paper's post-level Hit@k/mAP reproduction.
    """

    features = load_features(config.artifact_dir / "features.npz")
    check_feature_config(features, config.features)
    features = select_symptom_channels(
        features,
        config.experiment.symptom_keys,
    )
    features = crop_series(features, config.features.sequence_length)
    mask = (
        np.arange(features.symptom_scores.shape[2])[None, :]
        < features.lengths[:, None]
    )
    relevance = np.einsum(
        "nkt,nt->nk",
        features.symptom_scores,
        mask.astype(np.float32),
    ) / features.lengths[:, None]
    truth = features.symptom_truth.astype(bool)
    eligible = truth.any(axis=1)
    rows = relevance[eligible]
    truth_rows = truth[eligible]
    if len(rows) == 0:
        raise ValueError("No users with released symptom indicators")

    hit1 = 0
    hit3 = 0
    average_precisions: list[float] = []
    for scores, labels in zip(rows, truth_rows):
        order = np.argsort(-scores)
        hit1 += int(labels[order[:1]].any())
        hit3 += int(labels[order[:3]].any())
        relevant = int(labels.sum())
        accumulated_precision = 0.0
        correct = 0
        for rank, index in enumerate(order, start=1):
            if labels[index]:
                correct += 1
                accumulated_precision += correct / rank
        average_precisions.append(accumulated_precision / relevant)

    aurocs: dict[str, float | None] = {}
    for index, name in enumerate(features.symptom_names):
        target = truth[:, index]
        if len(np.unique(target)) < 2:
            aurocs[name] = None
        else:
            aurocs[name] = float(
                roc_auc_score(target, relevance[:, index])
            )
    valid_aurocs = [value for value in aurocs.values() if value is not None]
    report = {
        "protocol_note": (
            "Derived from released user-level symptom indicators and "
            "length-masked mean relevance; not the paper's post-level metric."
        ),
        "users_evaluated": int(len(rows)),
        "symptoms": features.symptom_names,
        "hit_at_1": float(hit1 / len(rows)),
        "hit_at_3": float(hit3 / len(rows)),
        "map": float(np.mean(average_precisions)),
        "macro_auroc": (
            float(np.mean(valid_aurocs)) if valid_aurocs else None
        ),
        "per_symptom_auroc": aurocs,
    }
    output_path = config.run_dir / "symptom_alignment_user_level.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
