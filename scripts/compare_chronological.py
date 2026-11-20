"""Evaluate chronological TSCD-Net and controls on identical user splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tscd_net.config import Config, PROJECT_ROOT
from tscd_net.features import TSCDFeatures, load_features
from tscd_net.metrics import best_f1_threshold, binary_metrics
from tscd_net.models.registry import build_classifier
from tscd_net.models.chronological import _pool_chronological_windows
from tscd_net.splits import DatasetSplit, stratified_70_10_20


DEEP_MODELS = ("chronological", "chronological_no_gap", "inceptiontime_windows")
SUMMARY_MODELS = ("symptom_means", "symptom_statistics", "symptom_statistics_boosted")


def balanced_shared_users(features: TSCDFeatures, baseline_cache: Path) -> np.ndarray:
    with np.load(baseline_cache, allow_pickle=False) as baseline:
        shared = np.isin(features.user_ids, baseline["user_ids"])
    by_class = [np.flatnonzero(shared & (features.labels == label)) for label in (0, 1)]
    count = min(map(len, by_class))
    rng = np.random.default_rng(42)
    return np.sort(np.concatenate([rng.choice(indices, count, replace=False) for indices in by_class]))


def distribution_features(scores: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Summarize the observed scores without including padding."""
    rows = []
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for values, length in zip(scores, lengths):
        observed = values[:, :length]
        rows.append(np.concatenate((
            observed.mean(axis=1),
            observed.std(axis=1),
            np.quantile(observed, quantiles, axis=1).reshape(-1),
            np.abs(np.diff(observed, axis=1)).sum(axis=1) / max(length - 1, 1),
        )))
    return np.stack(rows)


def predict_summary(
    name: str, values: np.ndarray, labels: np.ndarray, split: DatasetSplit, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if name == "symptom_statistics_boosted":
        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=200, max_leaf_nodes=15, l2_regularization=2.0,
            class_weight="balanced", random_state=seed,
        )
        model.fit(values[split.train], labels[split.train])
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(values[split.train], labels[split.train])
    return (
        model.predict_proba(values[split.validation])[:, 1],
        model.predict_proba(values[split.test])[:, 1],
    )


def predict_chronological(
    name: str,
    config: Config,
    scores: np.ndarray,
    gaps: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    anchors: np.ndarray,
    split: DatasetSplit,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model_config = replace(config.model, temporal_encoding=name != "chronological_no_gap")
    model = build_classifier(
        "tscdnet_chronological", model_config, config.training, seed, anchors,
    )
    tr, va, te = split.train, split.validation, split.test
    model.fit(
        scores[tr], gaps[tr], lengths[tr], labels[tr],
        scores[va], gaps[va], lengths[va], labels[va],
    )
    return (
        model.predict_proba(scores[va], gaps[va], lengths[va]),
        model.predict_proba(scores[te], gaps[te], lengths[te]),
    )


def symptom_windows(
    scores: np.ndarray, gaps: np.ndarray, lengths: np.ndarray, windows: int,
) -> np.ndarray:
    pooled = []
    for start in range(0, len(scores), 256):
        block = slice(start, start + 256)
        window_scores, _, _ = _pool_chronological_windows(
            torch.from_numpy(scores[block]),
            torch.from_numpy(gaps[block]),
            torch.from_numpy(lengths[block]),
            windows,
        )
        pooled.append(window_scores.numpy())
    return np.concatenate(pooled)


def predict_inception(
    windows: np.ndarray, labels: np.ndarray, split: DatasetSplit, seed: int, config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, str(PROJECT_ROOT.parent / "depression detect" / "src"))
    from dsts_depression.config import ModelConfig, TrainingConfig
    from dsts_depression.models.torch_trainer import TorchTimeSeriesClassifier

    tr, va, te = split.train, split.validation, split.test
    train = windows[tr]
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True).clip(min=1e-6)
    normalized = (windows - mean) / std
    model = TorchTimeSeriesClassifier(
        ModelConfig(device=config.model.device),
        TrainingConfig(
            epochs=config.training.epochs, batch_size=config.training.batch_size,
            learning_rate=0.001, use_amp=config.training.use_amp,
        ),
        seed,
    )
    model.fit(normalized[tr], labels[tr], normalized[va], labels[va])
    return (
        model.predict_proba(normalized[va])[:, 1],
        model.predict_proba(normalized[te])[:, 1],
    )


def save_summary(path: Path, rows: list[dict]) -> None:
    metrics = ("accuracy", "precision", "recall", "f1", "f1_at_0_5", "average_precision")
    columns = ["model", "runs"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name in sorted({row["model"] for row in rows}):
            selected = [row for row in rows if row["model"] == name]
            summary = {"model": name, "runs": len(selected)}
            for metric in metrics:
                values = np.asarray([row[metric] for row in selected])
                summary[f"{metric}_mean"] = values.mean()
                summary[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else ""
            writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/chronological.json")
    parser.add_argument("--population", choices=("observed", "balanced"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=DEEP_MODELS + SUMMARY_MODELS, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    config = Config.from_json(args.config)
    config.training.epochs = args.epochs
    config.training.runs = args.runs
    config.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    features = load_features(config.artifact_dir / "features.npz")
    if args.population == "balanced":
        baseline_cache = PROJECT_ROOT.parent / "depression detect/artifacts/main/features.npz"
        cohort = balanced_shared_users(features, baseline_cache)
    else:
        cohort = slice(None)
    scores = features.symptom_scores[cohort]
    gaps = features.log_time_gaps[cohort]
    lengths = features.lengths[cohort]
    labels = features.labels[cohort]
    user_ids = features.user_ids[cohort]
    anchors = features.symptom_embeddings
    del features
    if "inceptiontime_windows" in args.models and lengths.min() < config.model.chronological_windows:
        raise ValueError("InceptionTime comparison needs at least one post in each window")
    stats = distribution_features(scores, lengths) if set(args.models).intersection(SUMMARY_MODELS) else None
    windows = (
        symptom_windows(scores, gaps, lengths, config.model.chronological_windows)
        if "inceptiontime_windows" in args.models else None
    )
    manifest = {
        "population": args.population,
        "users": len(labels),
        "positive": int(labels.sum()),
        "models": args.models,
        "config": config.to_dict(),
    }
    (args.output / "protocol.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows: list[dict] = []
    for seed in range(config.training.master_seed, config.training.master_seed + args.runs):
        split = stratified_70_10_20(labels, seed)
        va, te = split.validation, split.test
        for name in args.models:
            print(f"Seed {seed}, {name}", flush=True)
            if name in SUMMARY_MODELS:
                inputs = stats[:, :scores.shape[1]] if name == "symptom_means" else stats
                validation, test = predict_summary(name, inputs, labels, split, seed)
            elif name == "inceptiontime_windows":
                validation, test = predict_inception(windows, labels, split, seed, config)
            else:
                validation, test = predict_chronological(
                    name, config, scores, gaps, lengths, labels, anchors, split, seed,
                )
            threshold = best_f1_threshold(labels[va], validation)
            metrics = binary_metrics(labels[te], test >= threshold)
            row = {
                "model": name, "seed": seed, "test_users": len(te),
                "test_positive": int(labels[te].sum()), "threshold": threshold, **metrics,
                "f1_at_0_5": float(f1_score(labels[te], test >= 0.5)),
                "average_precision": float(average_precision_score(labels[te], test)),
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"{seed}_{name}_predictions.npz", threshold=threshold,
                validation_user_ids=user_ids[va], validation_labels=labels[va],
                validation_probabilities=validation, test_user_ids=user_ids[te],
                test_labels=labels[te], test_probabilities=test,
            )
            (args.output / "metrics_by_run.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            save_summary(args.output / "metrics_summary.csv", rows)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
