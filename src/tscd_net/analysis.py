from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import Config
from .features import check_feature_config, load_features, select_symptom_channels
from .preprocess import crop_series


def analyze_symptom_trajectories(config: Config) -> Path:
    features_path = config.artifact_dir / "features.npz"
    features = load_features(features_path)
    check_feature_config(features, config.features)
    features = select_symptom_channels(
        features,
        config.experiment.symptom_keys,
    )
    features = crop_series(features, config.features.sequence_length)
    output_dir = config.run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    masked_mean = _masked_user_means(features)
    _write_mean_summary(features, masked_mean, output_dir / "user_symptoms.csv")
    representatives = _representative_users(features, masked_mean)
    _write_trajectories(features, representatives, output_dir)
    print(
        "Representative users:",
        {
            int(label): features.user_ids[index]
            for label, index in representatives.items()
        },
    )
    return output_dir


def _masked_user_means(features) -> np.ndarray:
    time = np.arange(features.symptom_scores.shape[2])[None, :]
    mask = time < features.lengths[:, None]
    sums = np.einsum(
        "nkt,nt->nk",
        features.symptom_scores,
        mask.astype(np.float32),
    )
    return sums / features.lengths[:, None]


def _write_mean_summary(features, means: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "user_id",
                "label",
                "length",
                *features.symptom_names,
            ]
        )
        for index, user_id in enumerate(features.user_ids):
            writer.writerow(
                [
                    str(user_id),
                    int(features.labels[index]),
                    int(features.lengths[index]),
                    *[
                        f"{value:.6f}"
                        for value in means[index]
                    ],
                ]
            )


def _representative_users(features, means: np.ndarray) -> dict[int, int]:
    score = means.mean(axis=1)
    representatives: dict[int, int] = {}
    for label in (0, 1):
        candidates = np.flatnonzero(features.labels == label)
        target = float(np.median(score[candidates]))
        representatives[label] = int(
            candidates[np.argmin(np.abs(score[candidates] - target))]
        )
    return representatives


def _write_trajectories(
    features,
    representatives: dict[int, int],
    output_dir: Path,
) -> None:
    bins = 120
    rows: list[list[object]] = []
    plotted: dict[int, np.ndarray] = {}
    for label, user_index in representatives.items():
        trajectory = _resample_trajectory(
            features.symptom_scores[user_index],
            int(features.lengths[user_index]),
            bins,
        )
        plotted[label] = trajectory
        for symptom_index, symptom_name in enumerate(features.symptom_names):
            for bin_index, value in enumerate(trajectory[symptom_index]):
                rows.append(
                    [
                        int(label),
                        str(features.user_ids[user_index]),
                        symptom_name,
                        bin_index,
                        f"{float(value):.6f}",
                    ]
                )
    with (output_dir / "representative_trajectories.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["label", "user_id", "symptom", "bin", "score"]
        )
        writer.writerows(rows)
    _try_plot(plotted, features.symptom_names, output_dir)


def _resample_trajectory(
    scores: np.ndarray,
    length: int,
    bins: int,
) -> np.ndarray:
    observed = scores[:, :length]
    assignments = np.minimum(
        (np.arange(length) * bins) // length,
        bins - 1,
    )
    output = np.zeros((len(scores), bins), dtype=np.float32)
    for bin_index in range(bins):
        members = assignments == bin_index
        if np.any(members):
            output[:, bin_index] = observed[:, members].mean(axis=1)
        elif bin_index > 0:
            output[:, bin_index] = output[:, bin_index - 1]
    return output


def _try_plot(
    trajectories: dict[int, np.ndarray],
    symptom_names: list[str],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    selected = [
        name for name in ("sadness", "anhedonia") if name in symptom_names
    ]
    if not selected:
        selected = symptom_names[:2]
    fig, axes = plt.subplots(1, len(selected), figsize=(11, 3.5))
    if len(selected) == 1:
        axes = [axes]
    for axis, symptom in zip(axes, selected):
        index = symptom_names.index(symptom)
        for label, color in ((0, "#4472c4"), (1, "#d62728")):
            name = "depressed" if label == 1 else "non-depressed"
            axis.plot(
                trajectories[label][index],
                color=color,
                linewidth=1.6,
                label=name,
            )
        axis.set_title(symptom)
        axis.set_xlabel("resampled bin")
        axis.set_ylabel("symptom score")
        axis.set_ylim(0.0, 1.0)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "symptom_trajectories.png", dpi=180)
    plt.close(fig)
