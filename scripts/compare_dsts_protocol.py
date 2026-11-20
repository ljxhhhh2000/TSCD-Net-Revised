"""Train the chronological TSCD-Net on the existing DSTS SWDD-7K splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import fields
from itertools import islice
from pathlib import Path

import numpy as np

from tscd_net.config import PROJECT_ROOT, Config
from tscd_net.features import TSCDFeatures, _log_time_gaps

BASELINE_ROOT = PROJECT_ROOT.parent / "depression detect"
sys.path.insert(0, str(BASELINE_ROOT / "src"))

from dsts_depression.config import Config as DSTSConfig, FeatureConfig
from dsts_depression.data import (
    _date_boundary,
    _extract_label,
    _make_record,
    _make_user_id,
    _open_streams,
)
from dsts_depression.features import DSTSExtractor, DSTSFeatures, load_features
from dsts_depression.splits import stratified_60_20_20


METRICS = ("accuracy", "precision", "recall", "f1")
_cleaner = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode(extractor: DSTSExtractor, texts: list[str]) -> np.ndarray:
    return extractor._get_encoder().encode(
        texts,
        batch_size=extractor.config.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)


def read_selected_records(baseline: DSTSConfig, dataset: Path, pending: dict[str, int]):
    start = _date_boundary(baseline.data.start_date)
    end = _date_boundary(baseline.data.end_date)
    with ExitStack() as stack:
        streams = _open_streams(stack, dataset, baseline.data.members)
        for member, stream in zip(baseline.data.members, streams):
            for line_number, line in enumerate(stream, start=1):
                payload = json.loads(line)
                user = payload.get("user")
                if not isinstance(user, dict):
                    continue
                user_id = _make_user_id(member, line_number, str(user.get("screen_name") or ""))
                index = pending.get(user_id)
                if index is None:
                    continue
                record = _make_record(
                    payload, _extract_label(payload), baseline.data, start, end, member, line_number,
                )
                pending.pop(user_id)
                yield index, record
    if pending:
        raise ValueError(f"{len(pending)} archived DSTS users were not found in the raw dataset")


def initialize_worker(config: FeatureConfig) -> None:
    global _cleaner
    _cleaner = DSTSExtractor(config).cleaner


def clean_alignment(job):
    index, record, expected_length, verify = job
    if len(record.texts) == expected_length and not verify:
        # The DSTS cleaner only drops posts. Equal counts imply every post survived.
        return index, _log_time_gaps(record.timestamps), None
    cleaned = [_cleaner(text) for text in record.texts]
    retained = [(text, time) for text, time in zip(cleaned, record.timestamps, strict=True) if text]
    if len(retained) != expected_length:
        raise ValueError(f"DSTS post count disagrees with the cache at row {index}")
    return index, _log_time_gaps([time for _, time in retained]), (
        [text for text, _ in retained] if verify else None
    )


def prepare_alignment(
    baseline: DSTSConfig, features: DSTSFeatures, dataset: Path,
    output: Path, cache_hash: str, workers: int,
) -> None:
    extractor = DSTSExtractor(baseline.features)
    keys = [symptom.key for symptom in extractor.symptoms]
    if keys != features.symptom_names:
        raise ValueError("The DSTS feature channels do not match their prompts")
    anchors = encode(extractor, [extractor.cleaner(s.description) for s in extractor.symptoms])
    width = features.values.shape[2]
    if baseline.data.max_tweets > width:
        raise ValueError("DSTS post limit exceeds the stored feature width")
    gaps = np.zeros((len(features.labels), width), dtype=np.float32)
    pending = {str(user_id): i for i, user_id in enumerate(features.user_ids)}
    sample_indices = set(np.random.default_rng(2023).choice(len(pending), 8, replace=False))
    max_score_error = 0.0
    records = read_selected_records(baseline, dataset, pending)
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_worker, initargs=(baseline.features,),
    ) as pool:
        while batch := list(islice(records, 256)):
            jobs = []
            for index, record in batch:
                if record.label != features.labels[index]:
                    raise ValueError(f"DSTS label disagrees with the cache at row {index}")
                jobs.append((index, record, int(features.lengths[index]), index in sample_indices))
            for index, user_gaps, texts in pool.map(clean_alignment, jobs, chunksize=8):
                gaps[index, :len(user_gaps)] = user_gaps
                if texts is not None:
                    calculated = (encode(extractor, texts) @ anchors.T).T
                    cached = features.values[index, :, :len(user_gaps)]
                    np.testing.assert_allclose(calculated, cached, atol=0.002, rtol=0)
                    max_score_error = max(max_score_error, float(np.max(np.abs(calculated - cached))))
                completed += 1
                if completed % 1000 == 0:
                    print(f"Aligned {completed}/{len(features.labels)} users", flush=True)

    temp = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temp, user_ids=features.user_ids, lengths=features.lengths, source_cache_sha256=cache_hash,
        log_time_gaps=gaps, symptom_embeddings=anchors,
        verified_score_max_abs_error=max_score_error, verified_users=len(sample_indices),
    )
    temp.replace(output)
    print(f"Saved alignment for {len(features.labels)} users", flush=True)


def load_aligned_features(
    source: DSTSFeatures, path: Path, cache_hash: str,
) -> TSCDFeatures:
    valid = np.arange(source.values.shape[-1])[None, :] < source.lengths[:, None]
    scores = np.where(valid[:, None, :], 0.5 * (1 + source.values), 0.0).astype(np.float32)
    with np.load(path, allow_pickle=False) as archive:
        if archive["source_cache_sha256"].item() != cache_hash:
            raise ValueError("The DSTS feature cache changed. Re-run prepare.")
        np.testing.assert_array_equal(source.user_ids, archive["user_ids"])
        np.testing.assert_array_equal(source.lengths, archive["lengths"])
        result = TSCDFeatures(
            symptom_scores=scores, log_time_gaps=archive["log_time_gaps"],
            lengths=source.lengths, labels=source.labels, user_ids=source.user_ids,
            symptom_names=source.symptom_names, symptom_embeddings=archive["symptom_embeddings"],
            embedding_model=source.embedding_model,
            symptom_truth=np.zeros(scores.shape[:2], dtype=np.uint8),
        )
    result.validate()
    return result


def compare(
    config: Config, features: TSCDFeatures, baseline_rows: list[dict],
    cache_hash: str, baseline_cache: Path,
) -> None:
    from tscd_net.experiment import _run_fingerprint, _train_run, _write_json, _write_jsonl, _write_summary

    output = config.run_dir
    fingerprint = _run_fingerprint(config, features)
    _write_json(output / "run_config.json", config.to_dict())
    _write_json(
        output / "protocol.json",
        {
            "DSTS_feature_cache": str(baseline_cache),
            "DSTS_feature_cache_sha256": cache_hash,
            "users": len(features.labels),
            "positive": int(features.labels.sum()),
            "symptoms": features.symptom_names,
            "split_method": "dsts_depression.splits.stratified_60_20_20",
            "split_seeds": [row["seed"] for row in baseline_rows],
            "score_transform": "(1 + cosine) / 2 on observed posts",
            "positive_decision": "probability > 0.5",
            "checkpoint": "validation accuracy, then validation loss",
        },
    )
    model = config.experiment.models[0]
    rows = []
    deltas = []
    for run_index, baseline_row in enumerate(baseline_rows):
        seed = baseline_row["seed"]
        split = stratified_60_20_20(features.labels, seed)
        for field, subset in (("train", split.train), ("validation", split.validation), ("test", split.test)):
            if len(subset) != baseline_row[f"{field}_users"]:
                raise ValueError(f"DSTS split size disagrees with the archive for seed {seed}")
            if len(subset) != 2 * features.labels[subset].sum():
                raise ValueError(f"{field} split is not balanced for seed {seed}")
        name = f"run_{run_index:02d}_{model}"
        row_path = output / "run_results" / f"{name}.json"
        prediction_path = output / "predictions" / f"{name}.npz"
        cached = json.loads(row_path.read_text(encoding="utf-8")) if row_path.is_file() else {}
        if (
            cached.get("run_fingerprint") == fingerprint
            and cached.get("seed") == seed
            and prediction_path.is_file()
            and prediction_path.with_suffix(".pt").is_file()
        ):
            print(f"Reuse seed {seed}", flush=True)
            row = cached
        else:
            print(f"Run {run_index + 1}/{len(baseline_rows)} seed {seed}", flush=True)
            row = _train_run(config, features, split, model, run_index, seed, fingerprint)
            _write_json(row_path, row)
        rows.append(row)
        delta = {"seed": seed, "test_users": row["test_users"]}
        for metric in METRICS:
            delta[f"dsts_{metric}"] = baseline_row[metric]
            delta[f"revised_{metric}"] = row[metric]
            delta[f"delta_{metric}"] = row[metric] - baseline_row[metric]
        deltas.append(delta)
        _write_jsonl(output / "metrics_by_run.jsonl", rows)
        _write_summary(rows + baseline_rows[:len(rows)], output / "metrics_summary.csv")
        with (output / "paired_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(delta))
            writer.writeheader()
            writer.writerows(deltas)
        print(f"Seed {seed}: TSCD-Net F1 {row['f1']:.4f}, DSTS-IT F1 {baseline_row['f1']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "all"))
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT.parents[1] / "Dataset/SWDD")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/dsts_protocol")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    baseline_dir = BASELINE_ROOT / "runs/main"
    baseline = DSTSConfig.from_json(baseline_dir / "run_config.json")
    baseline_cache = baseline.artifact_dir / "features.npz"
    cache_hash = file_sha256(baseline_cache)
    source = load_features(baseline_cache)
    if np.bincount(source.labels, minlength=2).tolist() != [3500, 3500]:
        raise ValueError("Expected the archived DSTS SWDD-7K with 3,500 users in each class")
    if len(np.unique(source.user_ids)) != len(source.user_ids):
        raise ValueError("Duplicate user IDs in the DSTS feature cache")
    args.output.mkdir(parents=True, exist_ok=True)
    alignment = args.output / "dsts_alignment.npz"
    archived_alignment = PROJECT_ROOT / "results/dsts_protocol/dsts_alignment.npz"
    if (
        args.command != "prepare"
        and not alignment.exists()
        and archived_alignment.is_file()
        and args.dataset.resolve() == PROJECT_ROOT.parents[1] / "Dataset/SWDD"
    ):
        alignment = archived_alignment

    if args.command == "prepare" or (args.command == "all" and not alignment.exists()):
        prepare_alignment(baseline, source, args.dataset.resolve(), alignment, cache_hash, args.workers)
    if args.command == "prepare":
        load_aligned_features(source, alignment, cache_hash)
        return

    features = load_aligned_features(source, alignment, cache_hash)
    config = Config.from_json(PROJECT_ROOT / "configs/chronological.json")
    for field in fields(config.data):
        setattr(config.data, field.name, getattr(baseline.data, field.name, getattr(config.data, field.name)))
    config.data.dataset_zip = str(args.dataset.resolve())
    config.features.embedding_model = source.embedding_model
    config.features.symptom_set = baseline.features.symptom_set
    config.training.master_seed = baseline.training.master_seed
    config.training.runs = baseline.training.runs
    config.training.threshold_strategy = "fixed"
    config.training.checkpoint_metric = "accuracy"
    config.experiment.output_dir = str(args.output.resolve())
    config.experiment.artifact_dir = str(alignment.parent.resolve())
    config.validate()
    baseline_rows = [
        json.loads(line) for line in (baseline_dir / "metrics_by_run.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected = list(range(config.training.master_seed, config.training.master_seed + config.training.runs))
    if [row["seed"] for row in baseline_rows] != expected:
        raise ValueError("Saved baseline results do not match the DSTS seed schedule")
    compare(config, features, baseline_rows, cache_hash, baseline_cache)


if __name__ == "__main__":
    main()
