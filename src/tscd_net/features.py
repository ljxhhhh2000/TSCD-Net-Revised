from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cleaning import TextCleaner
from .config import FeatureConfig
from .data import UserRecord
from .symptoms import get_symptoms, symptom_names


@dataclass
class UserFeature:
    symptom_scores: np.ndarray
    log_time_gaps: np.ndarray
    length: int
    label: int
    symptom_truth: np.ndarray
    user_id: str


class InsufficientCleanPosts(ValueError):
    pass


@dataclass
class TSCDFeatures:
    symptom_scores: np.ndarray
    log_time_gaps: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray
    symptom_truth: np.ndarray
    user_ids: np.ndarray
    symptom_names: list[str]
    symptom_embeddings: np.ndarray
    embedding_model: str

    def validate(self) -> None:
        if self.symptom_scores.ndim != 3:
            raise ValueError(
                "symptom_scores must have shape [users, symptoms, time]"
            )
        users, channels, time = self.symptom_scores.shape
        if self.log_time_gaps.shape != (users, time):
            raise ValueError("log_time_gaps shape does not match scores")
        if len(self.lengths) != users or len(self.labels) != users:
            raise ValueError("Users, lengths, and labels are not aligned")
        if len(self.user_ids) != users:
            raise ValueError("user_ids are not aligned with users")
        if self.symptom_truth.shape != (users, channels):
            raise ValueError("symptom_truth shape does not match scores")
        if self.symptom_embeddings.shape[0] != channels:
            raise ValueError("symptom_embeddings do not match channels")
        if len(self.symptom_names) != channels:
            raise ValueError("symptom_names do not match channels")
        if np.any(self.lengths < 1):
            raise ValueError("Every user must have at least one timestep")
        if np.any(self.lengths > time):
            raise ValueError("A sequence exceeds the padded width")
        if not np.all(np.isfinite(self.symptom_scores)):
            raise ValueError("symptom_scores contain non-finite values")
        if not np.all(np.isfinite(self.log_time_gaps)):
            raise ValueError("log_time_gaps contain non-finite values")
        if np.any(self.symptom_scores < 0.0) or np.any(
            self.symptom_scores > 1.0
        ):
            raise ValueError("symptom scores must lie in [0, 1]")


class TSCDFeatureExtractor:
    def __init__(
        self,
        config: FeatureConfig,
        min_posts: int = 1,
    ) -> None:
        self.config = config
        self.min_posts = min_posts
        self.symptoms = get_symptoms(config.symptom_set)
        self.cleaner = TextCleaner(
            drop_punctuation=config.drop_punctuation,
            convert_traditional=config.convert_traditional,
        )
        self._encoder = None
        self._symptom_embeddings: np.ndarray | None = None

    @property
    def symptom_embeddings(self) -> np.ndarray:
        if self._symptom_embeddings is None:
            encoder = self._get_encoder()
            descriptions = [
                self.cleaner(symptom.description) for symptom in self.symptoms
            ]
            self._symptom_embeddings = encoder.encode(
                descriptions,
                batch_size=self.config.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32, copy=False)
        return self._symptom_embeddings

    def extract_user(self, record: UserRecord) -> UserFeature:
        texts = [self.cleaner(text) for text in record.texts]
        pairs = [
            (text, timestamp)
            for text, timestamp in zip(texts, record.timestamps)
            if text
        ]
        if not pairs:
            raise ValueError(f"User {record.user_id} has no clean posts")
        if len(pairs) < self.min_posts:
            raise InsufficientCleanPosts(
                f"User {record.user_id} has only {len(pairs)} clean posts"
            )
        texts = [item[0] for item in pairs]
        timestamps = [item[1] for item in pairs]
        encoder = self._get_encoder()
        tweet_embeddings = encoder.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

        cosine = tweet_embeddings @ self.symptom_embeddings.T
        scores = ((1.0 + cosine) / 2.0).clip(0.0, 1.0)
        log_gaps = _log_time_gaps(timestamps)
        return UserFeature(
            symptom_scores=scores.T.astype(np.float32, copy=False),
            log_time_gaps=log_gaps,
            length=len(texts),
            label=record.label,
            symptom_truth=np.asarray(record.symptom_truth, dtype=np.uint8),
            user_id=record.user_id,
        )

    def write_checkpoint(self, feature: UserFeature, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            symptom_scores=feature.symptom_scores,
            log_time_gaps=feature.log_time_gaps,
            length=np.asarray(feature.length, dtype=np.int32),
            label=np.asarray(feature.label, dtype=np.int8),
            symptom_truth=feature.symptom_truth,
            user_id=np.asarray(feature.user_id),
        )

    def read_checkpoint(
        self,
        path: Path,
        expected_user_id: str,
        expected_label: int,
    ) -> UserFeature:
        with np.load(path, allow_pickle=False) as archive:
            feature = UserFeature(
                symptom_scores=archive["symptom_scores"].astype(
                    np.float32, copy=False
                ),
                log_time_gaps=archive["log_time_gaps"].astype(
                    np.float32, copy=False
                ),
                length=int(archive["length"]),
                label=int(archive["label"]),
                symptom_truth=archive["symptom_truth"].astype(
                    np.uint8, copy=False
                ),
                user_id=str(archive["user_id"]),
            )
        expected_shape = (
            len(self.symptoms),
            min(self.config.cache_length, feature.length),
        )
        if feature.symptom_scores.shape != expected_shape:
            raise ValueError(
                f"Checkpoint channel/time shape mismatch: {path}"
            )
        if feature.user_id != expected_user_id or feature.label != expected_label:
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        if feature.length < self.min_posts:
            raise InsufficientCleanPosts(
                f"Cached user {feature.user_id} has only "
                f"{feature.length} clean posts"
            )
        return feature

    def write_metadata(self, checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = checkpoint_dir / "metadata.json"
        expected = {
            "embedding_model": self.config.embedding_model,
            "symptom_names": symptom_names(self.symptoms),
            "descriptions": [
                symptom.description for symptom in self.symptoms
            ],
            "cache_length": self.config.cache_length,
            "drop_punctuation": self.config.drop_punctuation,
            "convert_traditional": self.config.convert_traditional,
            "score_transform": "(1 + cosine) / 2",
            "time_transform": "log(1 + days_between_adjacent_posts)",
        }
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing != expected:
                raise ValueError(
                    f"Feature checkpoint metadata mismatch: {metadata_path}"
                )
            return
        metadata_path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_encoder(self):
        if self._encoder is None:
            if self.config.embedding_model.startswith("hashing"):
                self._encoder = HashingTextEncoder()
            else:
                from sentence_transformers import SentenceTransformer

                device = self.config.embedding_device
                if device == "auto":
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                self._encoder = SentenceTransformer(
                    self.config.embedding_model,
                    device=device,
                )
                self._encoder.max_seq_length = 128
        return self._encoder


class HashingTextEncoder:
    """Deterministic encoder for tests and pipeline smoke checks only."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del batch_size, convert_to_numpy, show_progress_bar
        output = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for ngram_size in (1, 2, 3):
                for start in range(max(len(text) - ngram_size + 1, 0)):
                    gram = text[start : start + ngram_size]
                    digest = hashlib.blake2b(
                        gram.encode("utf-8"),
                        digest_size=8,
                    ).digest()
                    value = int.from_bytes(digest, "little")
                    output[row, value % self.dimensions] += (
                        1.0 if (value >> 63) & 1 else -1.0
                    )
            if normalize_embeddings:
                norm = float(np.linalg.norm(output[row]))
                if norm > 0:
                    output[row] /= norm
        return output


def assemble_features(
    user_features: list[UserFeature],
    extractor: TSCDFeatureExtractor,
    width: int,
) -> TSCDFeatures:
    if not user_features:
        raise ValueError("No eligible users were extracted")
    channels = len(extractor.symptoms)
    scores = np.zeros(
        (len(user_features), channels, width),
        dtype=np.float32,
    )
    gaps = np.zeros((len(user_features), width), dtype=np.float32)
    lengths = np.empty(len(user_features), dtype=np.int32)
    labels = np.empty(len(user_features), dtype=np.int8)
    symptom_truth = np.empty(
        (len(user_features), channels),
        dtype=np.uint8,
    )
    user_ids = np.empty(len(user_features), dtype="<U40")
    for index, feature in enumerate(user_features):
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
        symptom_names=symptom_names(extractor.symptoms),
        symptom_embeddings=extractor.symptom_embeddings,
        embedding_model=extractor.config.embedding_model,
    )
    result.validate()
    return result


def save_features(features: TSCDFeatures, path: Path) -> None:
    features.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        symptom_scores=features.symptom_scores,
        log_time_gaps=features.log_time_gaps,
        lengths=features.lengths,
        labels=features.labels,
        symptom_truth=features.symptom_truth,
        user_ids=features.user_ids,
        symptom_names=np.asarray(features.symptom_names),
        symptom_embeddings=features.symptom_embeddings,
        embedding_model=np.asarray(features.embedding_model),
    )
    metadata = {
        "shape": list(features.symptom_scores.shape),
        "users": int(len(features.labels)),
        "control": int(np.sum(features.labels == 0)),
        "depressed": int(np.sum(features.labels == 1)),
        "length_min": int(features.lengths.min()),
        "length_max": int(features.lengths.max()),
        "length_mean": float(features.lengths.mean()),
        "symptom_names": features.symptom_names,
        "embedding_model": features.embedding_model,
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_features(path: Path) -> TSCDFeatures:
    with np.load(path, allow_pickle=False) as archive:
        features = TSCDFeatures(
            symptom_scores=archive["symptom_scores"],
            log_time_gaps=archive["log_time_gaps"],
            lengths=archive["lengths"],
            labels=archive["labels"],
            symptom_truth=archive["symptom_truth"],
            user_ids=archive["user_ids"],
            symptom_names=[str(value) for value in archive["symptom_names"]],
            symptom_embeddings=archive["symptom_embeddings"],
            embedding_model=str(archive["embedding_model"]),
        )
    features.validate()
    return features


def check_feature_config(
    features: TSCDFeatures,
    config: FeatureConfig,
) -> None:
    expected_symptoms = symptom_names(get_symptoms(config.symptom_set))
    if features.embedding_model != config.embedding_model:
        raise ValueError(
            "Cached features use a different embedding model: "
            f"{features.embedding_model!r} != {config.embedding_model!r}"
        )
    if features.symptom_names != expected_symptoms:
        raise ValueError(
            "Cached symptom channels do not match the requested symptom set"
        )
    if features.symptom_scores.shape[2] != config.cache_length:
        raise ValueError("Cached feature width differs from cache_length")
    if features.symptom_scores.shape[2] < config.sequence_length:
        raise ValueError("Cached features are shorter than sequence_length")


def select_symptom_channels(
    features: TSCDFeatures,
    keys: list[str],
) -> TSCDFeatures:
    if not keys:
        return features
    unknown = set(keys) - set(features.symptom_names)
    if unknown:
        raise ValueError(f"Unknown symptom channels: {sorted(unknown)}")
    indices = [features.symptom_names.index(key) for key in keys]
    selected = TSCDFeatures(
        symptom_scores=features.symptom_scores[:, indices, :],
        log_time_gaps=features.log_time_gaps,
        lengths=features.lengths,
        labels=features.labels,
        symptom_truth=features.symptom_truth[:, indices],
        user_ids=features.user_ids,
        symptom_names=keys,
        symptom_embeddings=features.symptom_embeddings[indices],
        embedding_model=features.embedding_model,
    )
    selected.validate()
    return selected


def _log_time_gaps(timestamps: list) -> np.ndarray:
    gaps = np.zeros(len(timestamps), dtype=np.float64)
    for index in range(1, len(timestamps)):
        seconds = (
            timestamps[index] - timestamps[index - 1]
        ).total_seconds()
        gaps[index] = np.log1p(max(seconds / 86400.0, 0.0))
    return gaps.astype(np.float32)
