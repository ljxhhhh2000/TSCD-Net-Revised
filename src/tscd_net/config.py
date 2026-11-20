from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
T = TypeVar("T")


@dataclass
class DataConfig:
    dataset_zip: str = "../../Dataset/SWDD"
    members: list[str] = field(
        default_factory=lambda: ["control.jsonl", "depressed.jsonl"]
    )
    min_tweets: int = 50
    max_tweets: int = 500
    include_retweets: bool = True
    start_date: str = "2010-01-01"
    end_date: str = "2022-01-01"
    scan_limit_per_file: int | None = None
    max_users_per_class: int | None = None
    random_seed: int = 2026


@dataclass
class FeatureConfig:
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_device: str = "auto"
    embedding_batch_size: int = 64
    symptom_set: str = "paper15"
    sequence_length: int = 500
    cache_length: int = 500
    drop_punctuation: bool = False
    convert_traditional: bool = True


@dataclass
class ModelConfig:
    name: str = "tscdnet_chronological"
    device: str = "auto"
    model_dim: int = 32
    time_dim: int = 16
    attention_heads: int = 4
    attention_layers: int = 1
    ffn_multiplier: int = 2
    dropout: float = 0.1
    symptom_attention: bool = True
    temporal_attention: bool = True
    temporal_encoding: bool = True
    chronological_windows: int = 16


@dataclass
class TrainingConfig:
    runs: int = 5
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 6
    class_weighting: bool = True
    master_seed: int = 2026
    use_amp: bool = True
    threshold_strategy: str = "validation_f1"
    checkpoint_metric: str = "f1"


@dataclass
class ExperimentConfig:
    output_dir: str = "runs/chronological"
    artifact_dir: str = "../../Dataset/TSCD-Net-Features/main"
    models: list[str] = field(default_factory=lambda: ["tscdnet_chronological"])
    symptom_keys: list[str] = field(default_factory=list)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        config = cls(
            data=_dataclass_from_dict(DataConfig, payload.get("data", {})),
            features=_dataclass_from_dict(
                FeatureConfig, payload.get("features", {})
            ),
            model=_dataclass_from_dict(ModelConfig, payload.get("model", {})),
            training=_dataclass_from_dict(
                TrainingConfig, payload.get("training", {})
            ),
            experiment=_dataclass_from_dict(
                ExperimentConfig, payload.get("experiment", {})
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0 < self.data.min_tweets <= self.data.max_tweets:
            raise ValueError("Require 0 < data.min_tweets <= data.max_tweets")
        if self.data.max_tweets > self.features.cache_length:
            raise ValueError(
                "features.cache_length must be at least data.max_tweets"
            )
        if not 0 < self.features.sequence_length <= self.features.cache_length:
            raise ValueError(
                "Require 0 < features.sequence_length <= features.cache_length"
            )
        if self.data.max_users_per_class is not None and (
            self.data.max_users_per_class < 2
        ):
            raise ValueError("max_users_per_class must be at least 2")
        if self.data.scan_limit_per_file is not None and (
            self.data.scan_limit_per_file < 1
        ):
            raise ValueError("scan_limit_per_file must be positive")
        if self.model.model_dim % self.model.attention_heads != 0:
            raise ValueError(
                "model_dim must be divisible by attention_heads"
            )
        if self.model.time_dim < 2:
            raise ValueError("time_dim must be at least 2")
        if self.model.attention_layers < 1:
            raise ValueError("attention_layers must be positive")
        if self.model.chronological_windows < 2:
            raise ValueError("chronological_windows must be at least 2")
        if self.training.threshold_strategy not in {"fixed", "validation_f1"}:
            raise ValueError("threshold_strategy must be fixed or validation_f1")
        if self.training.checkpoint_metric not in {"f1", "accuracy"}:
            raise ValueError("checkpoint_metric must be f1 or accuracy")
        if self.training.runs < 1:
            raise ValueError("training.runs must be positive")
        if self.training.epochs < 1:
            raise ValueError("training.epochs must be positive")
        if self.training.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def dataset_path(self) -> Path:
        return _resolve_project_path(self.data.dataset_zip)

    @property
    def run_dir(self) -> Path:
        return _resolve_project_path(self.experiment.output_dir)

    @property
    def artifact_dir(self) -> Path:
        return _resolve_project_path(self.experiment.artifact_dir)


def _dataclass_from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys for {cls.__name__}: {sorted(unknown)}"
        )
    return cls(**payload)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
