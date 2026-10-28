from pathlib import Path

import pytest
import yaml

from deepvision.training.config import ConfigError, load_train_config

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def repo_cwd(monkeypatch):
    # data.yaml paths in train.yaml are relative to the repository root.
    monkeypatch.chdir(REPO_ROOT)


def project_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))


def write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_project_config_is_valid():
    config = load_train_config()

    assert config.seed == 42
    assert (config.model.backbone, config.model.pretrained, config.model.dropout) == (
        "resnet18",
        True,
        0.2,
    )
    training, fine_tune, stopping = (
        config.training,
        config.training.fine_tune,
        config.training.early_stopping,
    )
    assert (training.epochs, training.batch_size) == (10, 32)
    assert training.gradient_accumulation_steps == 1
    assert training.mixed_precision is True
    assert fine_tune.head_epochs == 3
    assert fine_tune.head_learning_rate == 0.001
    assert fine_tune.backbone_learning_rate == 0.0001
    assert (stopping.monitor, stopping.mode) == ("val_macro_f1", "max")
    assert config.mlflow.tracking_uri == "sqlite:///mlflow.db"
    assert config.checkpoint_dir == Path("artifacts/checkpoints")
    assert config.reports_dir == Path("reports/training")
    assert config.data.batch_size == training.batch_size
    assert len(config.data.class_names) == 6


def test_mlflow_database_is_ignored_by_git():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "mlflow.db" in gitignore


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("model", "backbone", "vgg16", "model.backbone"),
        ("model", "dropout", 1.5, "model.dropout"),
        ("training", "epochs", 0, "training.epochs"),
        ("training", "batch_size", 0, "training.batch_size"),
        ("training", "gradient_accumulation_steps", 0, "gradient_accumulation_steps"),
        ("training", "optimizer", "sgd", "training.optimizer"),
        ("training", "scheduler", "step", "training.scheduler"),
        ("training", "device", "tpu", "training.device"),
        ("fine_tune", "head_epochs", 11, "head_epochs"),
        ("fine_tune", "head_epochs", -1, "head_epochs"),
        ("fine_tune", "backbone_learning_rate", 0, "backbone_learning_rate"),
        ("early_stopping", "monitor", "test_accuracy", "early_stopping.monitor"),
        ("early_stopping", "mode", "min", "early_stopping.mode"),
        ("early_stopping", "patience", 0, "patience"),
        ("mlflow", "tracking_uri", "mlruns", "file store"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, section, key, value, message):
    config = project_config()
    nested = section in ("fine_tune", "early_stopping")
    target = config["training"][section] if nested else config[section]
    target[key] = value

    with pytest.raises(ConfigError, match=message):
        load_train_config(write(tmp_path, config))


def test_unknown_or_missing_keys_are_rejected(tmp_path):
    config = project_config()
    config["training"]["learning_rate"] = 0.01  # a key from the old layout
    with pytest.raises(ConfigError, match="learning_rate"):
        load_train_config(write(tmp_path, config))

    config = project_config()
    del config["training"]["fine_tune"]
    with pytest.raises(ConfigError, match="fine_tune"):
        load_train_config(write(tmp_path, config))


def test_head_only_and_full_fine_tune_extremes_are_allowed(tmp_path):
    for head_epochs in (0, 10):
        config = project_config()
        config["training"]["fine_tune"]["head_epochs"] = head_epochs
        loaded = load_train_config(write(tmp_path, config))
        assert loaded.training.fine_tune.head_epochs == head_epochs
