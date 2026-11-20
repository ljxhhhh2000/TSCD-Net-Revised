"""Fuse chronological TSCD-Net with the symptom-distribution control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from compare_chronological import save_summary
from tscd_net.config import PROJECT_ROOT
from tscd_net.metrics import best_f1_threshold, binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    run_dir = PROJECT_ROOT / "runs"
    parser.add_argument("--temporal", type=Path, default=run_dir / "chronological")
    parser.add_argument("--statistics", type=Path, default=run_dir / "chronological_controls")
    parser.add_argument("--output", type=Path, default=run_dir / "chronological_fusion")
    parser.add_argument("--paired", type=Path, help="Use predictions from a paired comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.paired is None:
        prediction_dir = args.temporal / "predictions"
        pairs = [
            (
                json.loads(result.read_text(encoding="utf-8"))["seed"],
                prediction_dir / result.with_suffix(".npz").name,
                args.statistics,
            )
            for result in sorted((args.temporal / "run_results").glob("run_*_tscdnet_chronological.json"))
        ]
    else:
        pairs = [
            (int(result.name.split("_", 1)[0]), result, args.paired)
            for result in sorted(args.paired.glob("*_chronological_predictions.npz"))
        ]
    if not pairs:
        raise FileNotFoundError("No completed chronological predictions")
    rows = []
    for seed, result, statistics_dir in pairs:
        with (
            np.load(result) as temporal,
            np.load(statistics_dir / f"{seed}_symptom_statistics_predictions.npz") as statistics,
        ):
            row = fuse(seed, temporal, statistics, args.output)
        rows.append(row)
    (args.output / "metrics_by_run.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_summary(args.output / "metrics_summary.csv", rows)


def fuse(seed: int, temporal, statistics, output: Path) -> dict:
    for key in ("validation_user_ids", "test_user_ids", "validation_labels", "test_labels"):
        if not np.array_equal(temporal[key], statistics[key]):
            raise ValueError(f"Seed {seed}: model outputs use different {key}")
    val_labels = temporal["validation_labels"]
    val_temporal = temporal["validation_probabilities"]
    val_statistics = statistics["validation_probabilities"]
    candidates = []
    for weight in (0.5, 0.25, 0.75, 0.0, 1.0):
        probabilities = weight * val_temporal + (1 - weight) * val_statistics
        threshold = best_f1_threshold(val_labels, probabilities)
        val_f1 = binary_metrics(val_labels, probabilities >= threshold)["f1"]
        candidates.append((val_f1, weight, threshold, probabilities))
    validation_f1, weight, threshold, val_probabilities = max(candidates, key=lambda item: item[0])
    test_labels = temporal["test_labels"]
    test_probabilities = (
        weight * temporal["test_probabilities"]
        + (1 - weight) * statistics["test_probabilities"]
    )
    np.savez_compressed(
        output / f"{seed}_fusion_predictions.npz",
        threshold=threshold, weight=weight,
        validation_user_ids=temporal["validation_user_ids"],
        validation_labels=val_labels, validation_probabilities=val_probabilities,
        test_user_ids=temporal["test_user_ids"],
        test_labels=test_labels, test_probabilities=test_probabilities,
    )
    return {
        "model": "validation_selected_fusion",
        "seed": seed, "temporal_weight": weight, "threshold": threshold,
        "validation_f1": validation_f1,
        "test_users": len(test_labels), "test_positive": int(test_labels.sum()),
        **binary_metrics(test_labels, test_probabilities >= threshold),
        "f1_at_0_5": binary_metrics(test_labels, test_probabilities >= 0.5)["f1"],
        "average_precision": float(average_precision_score(test_labels, test_probabilities)),
    }


if __name__ == "__main__":
    main()
