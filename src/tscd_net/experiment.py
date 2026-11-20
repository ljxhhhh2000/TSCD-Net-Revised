from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from .config import Config
from .data import iter_dataset
from .features import (
    InsufficientCleanPosts,
    TSCDFeatureExtractor,
    TSCDFeatures,
    check_feature_config,
    load_features,
    save_features,
    select_symptom_channels,
)
from .metrics import binary_metrics
from .models.registry import build_classifier
from .preprocess import crop_series
from .splits import stratified_70_10_20


@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    user_id: str
    label: int


def prepare(config: Config, force: bool = False) -> Path:
    artifact_dir = config.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    features_path = artifact_dir / "features.npz"
    cache_manifest_path = artifact_dir / "cache_manifest.json"
    fingerprint = _feature_fingerprint(config)

    if not force and features_path.is_file() and cache_manifest_path.is_file():
        cached = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            features = load_features(features_path)
            check_feature_config(features, config.features)
            print(f"Reusing feature cache: {features_path}")
            return features_path

    checkpoint_dir = (
        artifact_dir / "user_features" / fingerprint[:16]
    )
    extractor = TSCDFeatureExtractor(
        config.features,
        min_posts=config.data.min_tweets,
    )
    extractor.write_metadata(checkpoint_dir)
    refs: list[CheckpointRef] = []
    tweet_counts = 0
    records = iter_dataset(config.data, config.features.symptom_set)
    for index, record in enumerate(
        tqdm(records, desc="Extracting TSCD features", unit="user")
    ):
        checkpoint_path = (
            checkpoint_dir / f"{index:05d}-{record.user_id}.npz"
        )
        try:
            if checkpoint_path.is_file():
                feature = extractor.read_checkpoint(
                    checkpoint_path,
                    expected_user_id=record.user_id,
                    expected_label=record.label,
                )
            else:
                feature = extractor.extract_user(record)
                extractor.write_checkpoint(feature, checkpoint_path)
        except InsufficientCleanPosts:
            continue
        refs.append(
            CheckpointRef(
                path=checkpoint_path,
                user_id=record.user_id,
                label=record.label,
            )
        )
        tweet_counts += len(record.texts)

    features = _assemble_from_checkpoints(
        refs,
        extractor,
        width=config.features.cache_length,
    )
    save_features(features, features_path)
    _write_json(
        cache_manifest_path,
        {
            "fingerprint": fingerprint,
            "pipeline": "tscd-net-paper15-v1",
            "data": asdict(config.data),
            "features": asdict(config.features),
        },
    )
    _write_json(
        artifact_dir / "dataset_manifest.json",
        {
            "users": {
                "control": int(np.sum(features.labels == 0)),
                "depressed": int(np.sum(features.labels == 1)),
            },
            "total_users": int(len(features.labels)),
            "tweets": int(tweet_counts),
            "symptoms": features.symptom_names,
            "window": "most_recent",
            "include_retweets": config.data.include_retweets,
        },
    )
    print(f"Wrote TSCD feature cache: {features_path}")
    return features_path


def train(config: Config, features_path: Path | None = None) -> Path:
    if features_path is None:
        features_path = config.artifact_dir / "features.npz"
    if not features_path.is_file():
        raise FileNotFoundError(
            f"{features_path} does not exist; run prepare first"
        )
    features = load_features(features_path)
    check_feature_config(features, config.features)
    features = select_symptom_channels(
        features,
        config.experiment.symptom_keys,
    )
    features = crop_series(features, config.features.sequence_length)
    run_fingerprint = _run_fingerprint(config, features)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.run_dir / "run_config.json", config.to_dict())
    metrics_path = config.run_dir / "metrics_by_run.jsonl"
    result_dir = config.run_dir / "run_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    per_run_rows: list[dict[str, Any]] = []

    for run_index in range(config.training.runs):
        seed = config.training.master_seed + run_index
        split = stratified_70_10_20(features.labels, seed)
        for model_name in config.experiment.models:
            result_path = result_dir / f"run_{run_index:02d}_{model_name}.json"
            if result_path.is_file():
                row = json.loads(result_path.read_text(encoding="utf-8"))
                if row.get("run_fingerprint") == run_fingerprint:
                    print(
                        f"Skip completed run {run_index + 1}/"
                        f"{config.training.runs}: {model_name}"
                    )
                else:
                    print(
                        f"Re-run changed configuration "
                        f"{run_index + 1}/{config.training.runs}: {model_name}"
                    )
                    row = _train_run(
                        config,
                        features,
                        split,
                        model_name,
                        run_index,
                        seed,
                        run_fingerprint,
                    )
                    _write_json(result_path, row)
            else:
                print(f"Run {run_index + 1}/{config.training.runs}: {model_name}")
                row = _train_run(
                    config,
                    features,
                    split,
                    model_name,
                    run_index,
                    seed,
                    run_fingerprint,
                )
                _write_json(result_path, row)
            per_run_rows.append(row)
            _write_jsonl(metrics_path, per_run_rows)

    summary_path = config.run_dir / "metrics_summary.csv"
    _write_summary(per_run_rows, summary_path)
    return summary_path


def _train_run(
    config: Config,
    features: TSCDFeatures,
    split,
    model_name: str,
    run_index: int,
    seed: int,
    run_fingerprint: str,
) -> dict[str, Any]:
    classifier = build_classifier(
        model_name,
        config.model,
        config.training,
        seed=seed,
        symptom_embeddings=features.symptom_embeddings,
    )
    classifier.fit(
        train_scores=features.symptom_scores[split.train],
        train_gaps=features.log_time_gaps[split.train],
        train_lengths=features.lengths[split.train],
        train_labels=features.labels[split.train],
        validation_scores=features.symptom_scores[split.validation],
        validation_gaps=features.log_time_gaps[split.validation],
        validation_lengths=features.lengths[split.validation],
        validation_labels=features.labels[split.validation],
    )
    probabilities = classifier.predict_proba(
        features.symptom_scores[split.test],
        features.log_time_gaps[split.test],
        features.lengths[split.test],
    )
    predictions = classifier.classify(probabilities, classifier.threshold)
    test_labels = features.labels[split.test]
    metrics = binary_metrics(
        test_labels,
        predictions,
    )
    prediction_dir = config.run_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"run_{run_index:02d}_{model_name}.npz"
    validation_probabilities = classifier.predict_proba(
        features.symptom_scores[split.validation],
        features.log_time_gaps[split.validation],
        features.lengths[split.validation],
    )
    np.savez_compressed(
        prediction_path,
        threshold=classifier.threshold,
        positive_on_threshold=bool(classifier.classify(np.array([classifier.threshold]), classifier.threshold)[0]),
        test_user_ids=features.user_ids[split.test],
        test_labels=test_labels,
        test_probabilities=probabilities,
        validation_user_ids=features.user_ids[split.validation],
        validation_labels=features.labels[split.validation],
        validation_probabilities=validation_probabilities,
    )
    torch.save(
        {name: tensor.cpu() for name, tensor in classifier.model.state_dict().items()},
        prediction_path.with_suffix(".pt"),
    )
    metrics.update(
        threshold=classifier.threshold,
        f1_at_0_5=binary_metrics(test_labels, classifier.classify(probabilities, 0.5))["f1"],
        average_precision=float(average_precision_score(test_labels, probabilities)),
        true_positive=int(np.sum((test_labels == 1) & (predictions == 1))),
        false_positive=int(np.sum((test_labels == 0) & (predictions == 1))),
        true_negative=int(np.sum((test_labels == 0) & (predictions == 0))),
        false_negative=int(np.sum((test_labels == 1) & (predictions == 0))),
    )
    return {
        "run": run_index,
        "seed": seed,
        "model": model_name,
        "run_fingerprint": run_fingerprint,
        "train_users": int(len(split.train)),
        "validation_users": int(len(split.validation)),
        "test_users": int(len(split.test)),
        "train_positive": int(
            np.sum(features.labels[split.train] == 1)
        ),
        "test_positive": int(
            np.sum(features.labels[split.test] == 1)
        ),
        "best_epoch": int(classifier.best_epoch),
        **metrics,
    }


def _assemble_from_checkpoints(
    refs: list[CheckpointRef],
    extractor: TSCDFeatureExtractor,
    width: int,
) -> TSCDFeatures:
    if not refs:
        raise ValueError("No eligible users were found")
    channels = len(extractor.symptoms)
    scores = np.zeros((len(refs), channels, width), dtype=np.float32)
    gaps = np.zeros((len(refs), width), dtype=np.float32)
    lengths = np.empty(len(refs), dtype=np.int32)
    labels = np.empty(len(refs), dtype=np.int8)
    symptom_truth = np.empty((len(refs), channels), dtype=np.uint8)
    user_ids = np.empty(len(refs), dtype="<U40")
    for index, ref in enumerate(
        tqdm(refs, desc="Assembling feature cache", unit="user")
    ):
        feature = extractor.read_checkpoint(
            ref.path,
            expected_user_id=ref.user_id,
            expected_label=ref.label,
        )
        length = min(feature.length, width)
        scores[index, :, :length] = feature.symptom_scores[:, :length]
        gaps[index, :length] = feature.log_time_gaps[:length]
        lengths[index] = length
        labels[index] = feature.label
        symptom_truth[index] = feature.symptom_truth
        user_ids[index] = feature.user_id
    result = TSCDFeatures(
        symptom_scores=scores,
        log_time_gaps=gaps,
        lengths=lengths,
        labels=labels,
        symptom_truth=symptom_truth,
        user_ids=user_ids,
        symptom_names=[
            symptom.key for symptom in extractor.symptoms
        ],
        symptom_embeddings=extractor.symptom_embeddings,
        embedding_model=extractor.config.embedding_model,
    )
    result.validate()
    return result


def _feature_fingerprint(config: Config) -> str:
    dataset_path = config.dataset_path
    if dataset_path.is_dir():
        stats = {
            name: (dataset_path / name).stat()
            for name in config.data.members
            if (dataset_path / name).is_file()
        }
        dataset_size = sum(stat.st_size for stat in stats.values())
        dataset_mtime = max(
            (stat.st_mtime_ns for stat in stats.values()),
            default=None,
        )
    else:
        stat = dataset_path.stat() if dataset_path.is_file() else None
        dataset_size = stat.st_size if stat else None
        dataset_mtime = stat.st_mtime_ns if stat else None
    feature_values = asdict(config.features)
    feature_values.pop("sequence_length", None)
    payload = {
        "pipeline": "tscd-net-paper15-v1",
        "data": asdict(config.data),
        "features": feature_values,
        "dataset_size": dataset_size,
        "dataset_mtime_ns": dataset_mtime,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _run_fingerprint(config: Config, features: TSCDFeatures) -> str:
    digest = hashlib.sha256()
    config_values = config.to_dict()
    digest.update(
        json.dumps(config_values, sort_keys=True).encode("utf-8")
    )
    digest.update(features.labels.tobytes())
    digest.update(features.lengths.tobytes())
    digest.update(features.user_ids.tobytes())
    digest.update(
        np.asarray(features.symptom_scores.shape, dtype=np.int64).tobytes()
    )
    for array in (features.symptom_scores, features.log_time_gaps, features.symptom_embeddings):
        for chunk in np.array_split(array, 32):
            digest.update(np.ascontiguousarray(chunk))
    package_dir = Path(__file__).parent
    for source in sorted(package_dir.glob("*.py")) + sorted((package_dir / "models").glob("*.py")):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    metric_names = ["accuracy", "precision", "recall", "f1"]
    if all("average_precision" in row for row in rows):
        metric_names.extend(("f1_at_0_5", "average_precision"))
    fieldnames = ["model", "runs"]
    for metric in metric_names:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, model_rows in sorted(grouped.items()):
            output: dict[str, Any] = {
                "model": model_name,
                "runs": len(model_rows),
            }
            for metric in metric_names:
                values = np.asarray(
                    [row[metric] for row in model_rows],
                    dtype=float,
                )
                output[f"{metric}_mean"] = float(values.mean())
                output[f"{metric}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
            writer.writerow(output)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
