from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from ..config import ModelConfig, TrainingConfig
from ..metrics import best_f1_threshold
from .tscdnet import TSCDNet


class TSCDClassifier:
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        seed: int,
        symptom_embeddings: np.ndarray,
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.seed = seed
        self.device = _resolve_device(model_config.device)
        self.symptom_embeddings = torch.as_tensor(
            symptom_embeddings,
            dtype=torch.float32,
        )
        self.model = self._build_model().to(self.device)
        self.requires_validation = True
        self.use_amp = self.device.type == "cuda" and training_config.use_amp
        self.best_epoch = 0
        self.threshold = 0.5

    def _build_model(self) -> TSCDNet:
        if self.model_config.name != "tscdnet_chronological":
            raise ValueError(
                f"Unsupported model: {self.model_config.name!r}"
            )
        return TSCDNet(
            symptom_embeddings=self.symptom_embeddings,
            model_dim=self.model_config.model_dim,
            time_dim=self.model_config.time_dim,
            heads=self.model_config.attention_heads,
            layers=self.model_config.attention_layers,
            dropout=self.model_config.dropout,
            ffn_multiplier=self.model_config.ffn_multiplier,
            symptom_attention=self.model_config.symptom_attention,
            temporal_attention=self.model_config.temporal_attention,
            temporal_encoding=self.model_config.temporal_encoding,
            chronological_windows=self.model_config.chronological_windows,
        )

    def fit(
        self,
        train_scores: np.ndarray,
        train_gaps: np.ndarray,
        train_lengths: np.ndarray,
        train_labels: np.ndarray,
        validation_scores: np.ndarray,
        validation_gaps: np.ndarray,
        validation_lengths: np.ndarray,
        validation_labels: np.ndarray,
    ) -> "TSCDClassifier":
        _seed_everything(self.seed)
        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        positive_weight = _positive_class_weight(
            train_labels,
            enabled=self.training_config.class_weighting,
        )
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                positive_weight,
                dtype=torch.float32,
                device=self.device,
            )
        )
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        train_scores_tensor = _to_tensor(train_scores)
        train_gaps_tensor = _to_tensor(train_gaps)
        train_lengths_tensor = torch.as_tensor(train_lengths, dtype=torch.long)
        train_labels_tensor = _to_tensor(train_labels, dtype=torch.float32)
        validation_scores_tensor = _to_tensor(validation_scores)
        validation_gaps_tensor = _to_tensor(validation_gaps)
        validation_lengths_tensor = torch.as_tensor(
            validation_lengths,
            dtype=torch.long,
        )
        validation_labels_tensor = _to_tensor(
            validation_labels,
            dtype=torch.long,
        )
        generator = torch.Generator().manual_seed(self.seed)

        best_metric = -1.0
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        batch_size = self.training_config.batch_size

        for epoch in range(self.training_config.epochs):
            epoch_started = time.monotonic()
            self.model.train()
            permutation = torch.randperm(
                len(train_scores_tensor),
                generator=generator,
            )
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start : start + batch_size]
                batch_scores = train_scores_tensor[indices].to(self.device)
                batch_gaps = train_gaps_tensor[indices].to(self.device)
                batch_lengths = train_lengths_tensor[indices].to(self.device)
                batch_labels = train_labels_tensor[indices].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    logits = self.model(
                        batch_scores,
                        batch_gaps,
                        batch_lengths,
                    ).squeeze(-1)
                    loss = loss_function(logits, batch_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            probabilities, validation_loss = self._predict_with_loss(
                validation_scores_tensor,
                validation_gaps_tensor,
                validation_lengths_tensor,
                validation_labels_tensor,
                loss_function,
            )
            labels = validation_labels_tensor.numpy()
            threshold = (
                best_f1_threshold(labels, probabilities)
                if self.training_config.threshold_strategy == "validation_f1"
                else 0.5
            )
            predictions = self.classify(probabilities, threshold)
            validation_f1 = float(
                f1_score(
                    labels,
                    predictions,
                    pos_label=1,
                    zero_division=0,
                )
            )
            validation_accuracy = float(np.mean(labels == predictions))
            validation_metric = (
                validation_accuracy if self.training_config.checkpoint_metric == "accuracy"
                else validation_f1
            )
            improved = validation_metric > best_metric or (
                validation_metric == best_metric and validation_loss < best_loss
            )
            if improved:
                best_metric = validation_metric
                best_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch + 1
                self.threshold = threshold
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == self.training_config.epochs:
                elapsed = time.monotonic() - epoch_started
                print(
                    f"  epoch {epoch + 1}/{self.training_config.epochs} "
                    f"val_loss={validation_loss:.4f} "
                    f"val_f1={validation_f1:.4f} "
                    f"val_acc={validation_accuracy:.4f} "
                    f"time={elapsed:.1f}s"
                )
            patience = self.training_config.early_stopping_patience
            if patience > 0 and stale_epochs >= patience:
                print(
                    f"  early stopping after {stale_epochs} stale epochs; "
                    f"best epoch={self.best_epoch}"
                )
                break

        if best_state is None:
            raise RuntimeError("Training did not produce a checkpoint")
        self.model.load_state_dict(best_state)
        return self

    def classify(self, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        if self.training_config.checkpoint_metric == "accuracy" and threshold == 0.5:
            # Match binary argmax, which assigns an exact tie to class 0.
            return probabilities > threshold
        return probabilities >= threshold

    def predict_proba(
        self,
        scores: np.ndarray,
        gaps: np.ndarray,
        lengths: np.ndarray,
    ) -> np.ndarray:
        self.model.eval()
        score_tensor = _to_tensor(scores)
        gap_tensor = _to_tensor(gaps)
        length_tensor = torch.as_tensor(lengths, dtype=torch.long)
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(score_tensor), 128):
                logits = self.model(
                    score_tensor[start : start + 128].to(self.device),
                    gap_tensor[start : start + 128].to(self.device),
                    length_tensor[start : start + 128].to(self.device),
                )
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probabilities, axis=0).reshape(-1)

    def _predict_with_loss(
        self,
        scores: torch.Tensor,
        gaps: torch.Tensor,
        lengths: torch.Tensor,
        labels: torch.Tensor,
        loss_function: nn.Module,
    ) -> tuple[np.ndarray, float]:
        self.model.eval()
        logits_list = []
        with torch.inference_mode():
            for start in range(0, len(scores), 128):
                logits_list.append(
                    self.model(
                        scores[start : start + 128].to(self.device),
                        gaps[start : start + 128].to(self.device),
                        lengths[start : start + 128].to(self.device),
                    )
                )
        logits = torch.cat(logits_list, dim=0)
        loss = float(
            loss_function(
                logits.squeeze(-1),
                labels.to(self.device).float(),
            ).item()
        )
        return torch.sigmoid(logits).cpu().numpy().reshape(-1), loss


def _positive_class_weight(labels: np.ndarray, enabled: bool) -> float:
    if not enabled:
        return 1.0
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


def _to_tensor(
    values: np.ndarray,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(values), dtype=dtype)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
