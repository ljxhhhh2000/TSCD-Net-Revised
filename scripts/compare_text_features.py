"""Test whether post text adds signal beyond the frozen symptom scores."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import vstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from compare_chronological import save_summary
from tscd_net.cleaning import TextCleaner
from tscd_net.config import Config, PROJECT_ROOT
from tscd_net.data import iter_dataset
from tscd_net.metrics import best_f1_threshold, binary_metrics
from tscd_net.splits import stratified_70_10_20


TOPIC = re.compile(r"#[^#\r\n]{1,80}#")
DIAGNOSIS = re.compile(r"抑郁(?:症|障碍)?|躁郁(?:症)?|双相(?:情感)?障碍|depress(?:ion|ive|ed)", re.I)
HAN_TEXT = re.compile(r"[^\u3400-\u9fff]+")


def read_documents(config: Config, ids: np.ndarray):
    desired = set(ids.tolist())
    cleaner = TextCleaner(convert_traditional=config.features.convert_traditional)
    vectorizer = HashingVectorizer(
        analyzer="char", ngram_range=(2, 3), alternate_sign=False,
        n_features=2**19, norm=None, dtype=np.float32,
    )
    documents = []
    seen = []
    blocks = []
    for record in iter_dataset(config.data, config.features.symptom_set):
        if record.user_id not in desired:
            continue
        indices = np.linspace(0, len(record.texts) - 1, min(len(record.texts), 24), dtype=int)
        posts = [cleaner(TOPIC.sub(" ", record.texts[i])) for i in indices]
        documents.append(" ".join(HAN_TEXT.sub(" ", DIAGNOSIS.sub(" ", post[:160])) for post in posts))
        seen.append(record.user_id)
        if len(documents) == 256:
            blocks.append(vectorizer.transform(documents))
            documents.clear()
            if len(seen) % 2048 == 0:
                print(f"Vectorized {len(seen)}/{len(ids)} users", flush=True)
    if documents:
        blocks.append(vectorizer.transform(documents))
    row_by_id = {value: index for index, value in enumerate(seen)}
    if set(row_by_id) != desired or len(seen) != len(desired):
        raise ValueError("User records do not match the feature cache")
    return vstack(blocks, format="csr")[[row_by_id[value] for value in ids]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/chronological.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/text_features")
    parser.add_argument("--statistics", type=Path, default=PROJECT_ROOT / "runs/chronological_controls")
    args = parser.parse_args()
    config = Config.from_json(args.config)
    with np.load(config.artifact_dir / "features.npz") as data:
        user_ids, labels = data["user_ids"], data["labels"]
    counts = read_documents(config, user_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(config.training.master_seed, config.training.master_seed + config.training.runs):
        split = stratified_70_10_20(labels, seed)
        tr, va, te = split.train, split.validation, split.test
        tfidf = TfidfTransformer(sublinear_tf=True)
        train_counts = tfidf.fit_transform(counts[tr])
        model = LogisticRegression(solver="liblinear", C=16, class_weight="balanced", max_iter=500)
        model.fit(train_counts, labels[tr])
        val_text = model.predict_proba(tfidf.transform(counts[va]))[:, 1]
        test_text = model.predict_proba(tfidf.transform(counts[te]))[:, 1]
        with np.load(args.statistics / f"{seed}_symptom_statistics_predictions.npz") as statistics:
            np.testing.assert_array_equal(user_ids[va], statistics["validation_user_ids"])
            np.testing.assert_array_equal(user_ids[te], statistics["test_user_ids"])
            val_stats, test_stats = statistics["validation_probabilities"], statistics["test_probabilities"]
        candidates = []
        for weight in (0.25, 0.5, 0.75):
            probabilities = weight * val_text + (1 - weight) * val_stats
            threshold = best_f1_threshold(labels[va], probabilities)
            val_f1 = binary_metrics(labels[va], probabilities >= threshold)["f1"]
            candidates.append((val_f1, weight, threshold))
        _, weight, fusion_threshold = max(candidates, key=lambda item: item[0])
        results = (
            ("text_only", 1.0, best_f1_threshold(labels[va], val_text), val_text, test_text),
            ("text_symptoms", weight, fusion_threshold, weight * val_text + (1 - weight) * val_stats,
             weight * test_text + (1 - weight) * test_stats),
        )
        for name, weight, threshold, validation, test in results:
            row = {
                "model": name, "seed": seed, "text_weight": weight, "threshold": threshold,
                "test_users": len(te), "test_positive": int(labels[te].sum()),
                **binary_metrics(labels[te], test >= threshold),
                "f1_at_0_5": binary_metrics(labels[te], test >= 0.5)["f1"],
                "average_precision": float(average_precision_score(labels[te], test)),
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"{seed}_{name}_predictions.npz", threshold=threshold,
                validation_user_ids=user_ids[va], validation_labels=labels[va],
                validation_probabilities=validation, test_user_ids=user_ids[te],
                test_labels=labels[te], test_probabilities=test,
            )
            print(json.dumps(row), flush=True)
        (args.output / "metrics_by_run.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        save_summary(args.output / "metrics_summary.csv", rows)


if __name__ == "__main__":
    main()
