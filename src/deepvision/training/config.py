"""Typed, validated training configuration loaded from configs/train.yaml.

Data settings are delegated to deepvision.data.dataset.load_data_config. Invalid
values raise ConfigError before any data is loaded or weights are downloaded.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from deepvision.data.dataset import DataConfig, load_data_config
from deepvision.models.factory import SUPPORTED_BACKBONES

OPTIMIZERS = frozenset({"adamw"})
SCHEDULERS = frozenset({"cosine", "none"})
DEVICES = frozenset({"auto", "cpu", "cuda"})
# Metric -> the direction in which it improves.
MONITORS = {"val_macro_f1": "max", "val_accuracy": "max", "val_loss": "min"}


class ConfigError(ValueError):
    """Raised when the training configuration is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    backbone: str
    pretrained: bool
    dropout: float


@dataclass(frozen=True)
class FineTuneConfig:
    head_epochs: int
    head_learning_rate: float
    backbone_learning_rate: float


@dataclass(frozen=True)
class EarlyStoppingConfig:
    monitor: str
    mode: str
    patience: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    mixed_precision: bool
    optimizer: str
    weight_decay: float
    scheduler: str
    device: str
    fine_tune: FineTuneConfig
    early_stopping: EarlyStoppingConfig


@dataclass(frozen=True)
class MlflowConfig:
    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    checkpoint_dir: Path
    mlflow: MlflowConfig
    reports_dir: Path = Path("reports/training")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def validate(config: TrainConfig) -> None:
    model, training = config.model, config.training
    fine_tune, stopping = training.fine_tune, training.early_stopping
    _require(
        model.backbone in SUPPORTED_BACKBONES,
        f"model.backbone {model.backbone!r} is not one of {sorted(SUPPORTED_BACKBONES)}",
    )
    _require(0 <= model.dropout < 1, f"model.dropout must be in [0, 1), got {model.dropout}")
    _require(training.epochs >= 1, "training.epochs must be at least 1")
    _require(training.batch_size >= 1, "training.batch_size must be at least 1")
    _require(
        training.gradient_accumulation_steps >= 1,
        "training.gradient_accumulation_steps must be at least 1",
    )
    _require(training.weight_decay >= 0, "training.weight_decay must not be negative")
    _require(training.optimizer in OPTIMIZERS, f"training.optimizer must be one of {OPTIMIZERS}")
    _require(training.scheduler in SCHEDULERS, f"training.scheduler must be one of {SCHEDULERS}")
    _require(training.device in DEVICES, f"training.device must be one of {sorted(DEVICES)}")
    _require(
        0 <= fine_tune.head_epochs <= training.epochs,
        f"fine_tune.head_epochs must be between 0 and epochs ({training.epochs}), "
        f"got {fine_tune.head_epochs}",
    )
    for name in ("head_learning_rate", "backbone_learning_rate"):
        _require(getattr(fine_tune, name) > 0, f"fine_tune.{name} must be positive")
    _require(
        stopping.monitor in MONITORS,
        f"early_stopping.monitor must be one of {sorted(MONITORS)}",
    )
    _require(
        stopping.mode == MONITORS.get(stopping.monitor),
        f"early_stopping.mode for {stopping.monitor} must be {MONITORS.get(stopping.monitor)!r}",
    )
    _require(stopping.patience >= 1, "early_stopping.patience must be at least 1")
    _require(
        "://" in config.mlflow.tracking_uri,
        f"mlflow.tracking_uri {config.mlflow.tracking_uri!r} is a file store path; MLflow 3 "
        "rejects it, use a database URI such as sqlite:///mlflow.db",
    )
    _require(bool(config.mlflow.experiment_name), "mlflow.experiment_name must not be empty")


def load_train_config(path: Path = Path("configs/train.yaml")) -> TrainConfig:
    """Load and validate the training configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        training = raw["training"]
        config = TrainConfig(
            seed=raw["seed"],
            data=load_data_config(path),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(
                **{k: v for k, v in training.items() if k not in ("fine_tune", "early_stopping")},
                fine_tune=FineTuneConfig(**training["fine_tune"]),
                early_stopping=EarlyStoppingConfig(**training["early_stopping"]),
            ),
            checkpoint_dir=Path(raw["output"]["checkpoint_dir"]),
            mlflow=MlflowConfig(**raw["mlflow"]),
            reports_dir=Path(raw["output"]["reports_dir"]),
        )
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"invalid training configuration {path}: {exc!r}") from exc
    validate(config)
    return config
