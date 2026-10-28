import contextlib
import csv
import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch
import torchvision
import yaml
from mlflow.tracking import MlflowClient
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import functional as F

from deepvision.evaluation import evaluate
from deepvision.evaluation.evaluate import (
    load_evaluation_config,
    main,
    mlflow_metrics,
)
from deepvision.evaluation.metrics import (
    bootstrap_intervals,
    expected_calibration_error,
    log_loss,
    normalize_rows,
    per_class_metrics,
    summarize,
    top_k_accuracy,
)
from deepvision.training import train
from deepvision.training.config import ConfigError
from deepvision.training.engine import confusion_matrix, load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["a", "b", "c"]
SPLIT_COUNTS = {"train": 8, "val": 3, "test": 3}


# --- Metrics ------------------------------------------------------------------------


def test_per_class_metrics_match_scikit_learn():
    generator = torch.Generator().manual_seed(0)
    targets = torch.randint(0, 4, (300,), generator=generator)
    predictions = torch.randint(0, 3, (300,), generator=generator)  # class 3 never predicted
    rows = per_class_metrics(confusion_matrix(targets, predictions, 4), list("wxyz"))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=range(4), zero_division=0
    )

    for i, row in enumerate(rows):
        assert row["class"] == "wxyz"[i] and row["support"] == support[i]
        assert (row["precision"], row["recall"], row["f1"]) == pytest.approx(
            (precision[i], recall[i], f1[i])
        )


def test_row_normalization_and_top_k():
    matrix = torch.tensor([[3, 1], [0, 0]])
    assert normalize_rows(matrix).tolist() == [[0.75, 0.25], [0.0, 0.0]]

    probabilities = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])
    targets = torch.tensor([1, 2, 0])
    assert top_k_accuracy(probabilities, targets, 1) == pytest.approx(1 / 3)
    assert top_k_accuracy(probabilities, targets, 2) == pytest.approx(2 / 3)
    assert top_k_accuracy(probabilities, targets, 5) == 1.0


def test_log_loss_matches_cross_entropy():
    logits = torch.randn(50, 6, generator=torch.Generator().manual_seed(1))
    targets = torch.randint(0, 6, (50,), generator=torch.Generator().manual_seed(2))
    expected = F.cross_entropy(logits.double(), targets).item()
    assert log_loss(torch.softmax(logits.double(), dim=1), targets) == pytest.approx(expected)


def test_expected_calibration_error():
    certain = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(certain, torch.tensor([0, 1])) == pytest.approx(0.0)

    overconfident = torch.tensor([[0.9, 0.1]] * 4)  # 90% confident, 50% correct
    assert expected_calibration_error(overconfident, torch.tensor([0, 0, 1, 1])) == pytest.approx(
        0.4
    )
    mixed = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
    # bin at 0.9: accuracy 1.0 (gap 0.1); bin at 0.6: accuracy 0.5 (gap 0.1)
    assert expected_calibration_error(mixed, torch.tensor([0, 0, 0, 1])) == pytest.approx(0.1)


def test_bootstrap_intervals_are_reproducible_and_bracket_the_estimate():
    generator = torch.Generator().manual_seed(3)
    targets = torch.randint(0, 3, (400,), generator=generator)
    predictions = torch.where(
        torch.rand(400, generator=generator) < 0.8, targets, (targets + 1) % 3
    )

    first = bootstrap_intervals(targets, predictions, 3, samples=200, seed=7)
    again = bootstrap_intervals(targets, predictions, 3, samples=200, seed=7)
    other = bootstrap_intervals(targets, predictions, 3, samples=200, seed=8)

    accuracy = (targets == predictions).double().mean().item()
    assert first == again and first != other
    assert first["accuracy"]["low"] < accuracy < first["accuracy"]["high"]
    assert 0 < first["macro_f1"]["low"] < first["macro_f1"]["high"] < 1


def test_summarize_headline_and_optional_intervals():
    probabilities = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]]
    )
    targets = torch.tensor([0, 1, 2, 1])

    result = summarize(probabilities, targets, CLASSES, bootstrap_samples=50, bootstrap_seed=0)
    headline = result["headline"]
    assert headline["images"] == 4 and headline["accuracy"] == pytest.approx(0.75)
    assert headline["top2_accuracy"] == 1.0
    assert set(headline["macro_f1_ci95"]) == {"low", "high"}
    assert result["confusion_matrix"].sum() == 4 and len(result["per_class"]) == 3

    plain = summarize(probabilities, targets, CLASSES, bootstrap_samples=0)
    assert "macro_f1_ci95" not in plain["headline"]


def test_mlflow_metric_names():
    metrics = {
        "test": {"macro_f1": 0.9, "macro_f1_ci95": {"low": 0.88, "high": 0.92}, "images": 9},
        "test_official": {"accuracy": 0.8},
        "validation_check": {"max_difference": 0.0},
    }
    assert mlflow_metrics(metrics) == {
        "test_macro_f1": 0.9,
        "test_macro_f1_ci95_low": 0.88,
        "test_macro_f1_ci95_high": 0.92,
        "test_images": 9.0,
        "test_official_accuracy": 0.8,
    }


# --- Configuration ------------------------------------------------------------------


def project_eval_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "evaluate.yaml").read_text(encoding="utf-8"))


def test_project_evaluation_config_is_valid():
    config = load_evaluation_config(REPO_ROOT / "configs" / "evaluate.yaml")
    assert config.checkpoint == Path("artifacts/checkpoints/best.pt")
    assert config.splits == ("test", "test_official")
    assert (config.batch_size, config.validation_check_tolerance) == (64, 0.005)
    assert (config.bootstrap_samples, config.bootstrap_seed, config.top_errors) == (1000, 42, 30)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"splits": ["test_official"]}, "must include test"),
        ({"splits": ["test", "val"]}, "distinct values"),
        ({"splits": ["test", "test"]}, "distinct values"),
        ({"batch_size": 0}, "batch_size"),
        ({"device": "tpu"}, "device"),
        ({"validation_check_tolerance": -0.1}, "tolerance"),
        ({"bootstrap": {"samples": -1, "seed": 1}}, "bootstrap.samples"),
        ({"top_errors": -1}, "top_errors"),
        ({"extra": 1}, "unknown \\['extra'\\]"),
    ],
)
def test_invalid_evaluation_config_is_rejected(tmp_path, change, message):
    path = tmp_path / "evaluate.yaml"
    path.write_text(yaml.safe_dump(project_eval_config() | change))
    with pytest.raises(ConfigError, match=message):
        load_evaluation_config(path)


def test_missing_evaluation_key_is_rejected(tmp_path):
    raw = project_eval_config()
    del raw["top_errors"]
    path = tmp_path / "evaluate.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="missing \\['top_errors'\\]"):
        load_evaluation_config(path)


# --- End to end ---------------------------------------------------------------------


def build_project(root: Path) -> str:
    """Create images for every split type, the manifest, and configs; return the MLflow URI."""
    rng = np.random.default_rng(0)
    rows = []

    def add(relpath: str, label: str, split: str, class_index: int) -> None:
        pixels = rng.integers(0, 120, (32, 32, 3), dtype=np.uint8)
        pixels[..., 0] += np.uint8(60 * class_index)
        path = root / "data" / "images" / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        rows.append({"relpath": relpath, "label": label, "split": split, "group_id": relpath})

    for index, label in enumerate(CLASSES):
        for split, count in SPLIT_COUNTS.items():
            source = "test" if split == "test" else "train"
            for i in range(count):
                add(f"{source}/{label}/{split}{i}.png", label, split, index)
    add("test/a/conflict.png", "a", "test_excluded", 0)
    add("train/b/excluded.png", "b", "excluded", 1)
    for i in range(2):
        add(f"pred/{i}.png", "", "pred", i)
    with (root / "data" / "splits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relpath", "label", "split", "group_id"])
        writer.writeheader()
        writer.writerows(rows)

    (root / "configs").mkdir()
    data = {"archive": {"output": "data/images"}, "split": {"output": "data/splits.csv"}}
    (root / "configs" / "data.yaml").write_text(yaml.safe_dump(data))
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    config = {
        "seed": 0,
        "data": {
            "config": "configs/data.yaml", "class_names": CLASSES, "num_classes": 3,
            "image_size": 32, "eval_batch_size": 4, "num_workers": 0, "pin_memory": False,
        },
        "model": {"backbone": "mobilenet_v3_small", "pretrained": False, "dropout": 0.0},
        "training": {
            "epochs": 2, "batch_size": 4, "gradient_accumulation_steps": 1,
            "mixed_precision": True, "optimizer": "adamw", "weight_decay": 0.0,
            "scheduler": "cosine", "device": "cpu",
            "fine_tune": {
                "head_epochs": 1, "head_learning_rate": 0.01, "backbone_learning_rate": 0.001,
            },
            "early_stopping": {"monitor": "val_macro_f1", "mode": "max", "patience": 3},
        },
        "output": {"checkpoint_dir": "artifacts/checkpoints", "reports_dir": "reports/training"},
        "mlflow": {"tracking_uri": uri, "experiment_name": "test-experiment"},
    }  # fmt: skip
    (root / "configs" / "train.yaml").write_text(yaml.safe_dump(config))
    return uri


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> tuple[Path, str]:
    """Train a tiny checkpoint once (with MLflow) for all end-to-end tests."""
    root = tmp_path_factory.mktemp("project")
    uri = build_project(root)
    with contextlib.chdir(root):
        assert train.main(["--config", "configs/train.yaml"]) == 0
    while mlflow.active_run() is not None:
        mlflow.end_run()
    return root, uri


@pytest.fixture
def project(trained, tmp_path, monkeypatch):
    """Per-test evaluation config pointing at a private copy of the checkpoint."""
    root, uri = trained
    monkeypatch.chdir(root)
    checkpoint = tmp_path / "best.pt"
    shutil.copyfile(root / "artifacts" / "checkpoints" / "best.pt", checkpoint)
    config = project_eval_config() | {
        "checkpoint": checkpoint.as_posix(),
        "reports_dir": (tmp_path / "reports").as_posix(),
        "batch_size": 4,
        "device": "cpu",
        "validation_check_tolerance": 1e-9,  # CPU training validated in FP32 too
        "bootstrap": {"samples": 100, "seed": 42},
        "top_errors": 5,
    }
    config_path = tmp_path / "evaluate.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return {"root": root, "uri": uri, "config": config_path, "checkpoint": checkpoint,
            "reports": tmp_path / "reports"}  # fmt: skip


def run(project, *args) -> int:
    return main(["--config", str(project["config"]), *args])


def edit_checkpoint(path: Path, edit) -> None:
    checkpoint = load_checkpoint(path)
    edit(checkpoint)
    torch.save(checkpoint, path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.integration
def test_evaluation_writes_reports_and_logs_to_the_training_run(project, capsys):
    assert run(project) == 0

    out = capsys.readouterr().out
    assert "validation check passed" in out and "test (9 images)" in out
    reports = project["reports"]
    metrics = json.loads((reports / "metrics.json").read_text())
    assert metrics["validation_check"]["max_difference"] < 1e-9
    assert (metrics["test"]["images"], metrics["test_official"]["images"]) == (9, 10)
    test = metrics["test"]
    assert test["macro_f1_ci95"]["low"] <= test["macro_f1"] <= test["macro_f1_ci95"]["high"]
    assert metrics["checkpoint"]["md5"] == train.file_md5(project["checkpoint"])
    assert metrics["checkpoint"]["epoch"] in (1, 2)

    predictions = read_csv(reports / "predictions.csv")
    paths = [row["relpath"] for row in predictions]
    assert paths == sorted(paths) and len(paths) == 9
    assert all(path.startswith("test/") and "conflict" not in path for path in paths)
    for row in predictions:
        assert sum(float(row[f"prob_{c}"]) for c in CLASSES) == pytest.approx(1, abs=1e-5)
        assert row["correct"] == str(row["label"] == row["predicted"])
    assert [row["relpath"] for row in read_csv(reports / "excluded_predictions.csv")] == [
        "test/a/conflict.png"
    ]
    errors = read_csv(reports / "errors.csv")
    assert len(errors) <= 5 and all(row["label"] != row["predicted"] for row in errors)
    confidences = [float(row["confidence"]) for row in errors]
    assert confidences == sorted(confidences, reverse=True)
    per_class = read_csv(reports / "per_class.csv")
    assert [(r["split"], r["class"]) for r in per_class] == [
        (s, c) for s in ("test", "test_official") for c in CLASSES
    ]
    matrix = read_csv(reports / "confusion_matrix.csv")
    assert [row["label"] for row in matrix] == CLASSES
    assert sum(int(row[c]) for row in matrix for c in CLASSES) == 9

    client = MlflowClient(project["uri"])
    checkpoint = load_checkpoint(project["checkpoint"])
    mlflow_run = client.get_run(checkpoint["metadata"]["mlflow_run_id"])
    assert mlflow_run.data.metrics["test_macro_f1"] == pytest.approx(test["macro_f1"])
    assert "test_official_macro_f1" in mlflow_run.data.metrics
    assert mlflow_run.data.tags["evaluated"] == "true"
    artifacts = {a.path for a in client.list_artifacts(mlflow_run.info.run_id, "evaluation")}
    assert "evaluation/metrics.json" in artifacts and "evaluation/predictions.csv" in artifacts


@pytest.mark.integration
def test_tampered_checkpoint_stops_before_the_test_set(project, monkeypatch, capsys):
    edit_checkpoint(project["checkpoint"], lambda ck: ck["metrics"].update(macro_f1=0.123))
    requested = []
    real = evaluate.build_dataloaders

    def spy(config, splits, **kwargs):
        requested.append(tuple(splits))
        return real(config, splits, **kwargs)

    monkeypatch.setattr(evaluate, "build_dataloaders", spy)
    assert run(project, "--no-mlflow") == 1

    assert "test set was not evaluated" in capsys.readouterr().err
    assert requested == [("val",)]
    assert not project["reports"].exists()


@pytest.mark.integration
def test_mismatched_classes_and_missing_checkpoint_are_rejected(project, capsys):
    edit_checkpoint(project["checkpoint"], lambda ck: ck.update(class_names=["x", "y", "z"]))
    assert run(project, "--no-mlflow") == 1
    assert "differ from the configured class_names" in capsys.readouterr().err

    project["checkpoint"].unlink()
    assert run(project, "--no-mlflow") == 1
    assert "dvc repro train" in capsys.readouterr().err


@pytest.mark.integration
def test_model_is_rebuilt_without_pretrained_weights(project, monkeypatch):
    requested = []
    real = torchvision.models.get_model

    def spy(name, weights=None):
        requested.append(weights)
        return real(name, weights=weights)

    monkeypatch.setattr(torchvision.models, "get_model", spy)
    assert run(project, "--no-mlflow") == 0
    assert requested == [None]


@pytest.mark.integration
def test_reports_are_deterministic(project, tmp_path):
    first = project["reports"]
    assert run(project, "--no-mlflow") == 0
    second = tmp_path / "again"
    config = yaml.safe_load(project["config"].read_text()) | {"reports_dir": second.as_posix()}
    project["config"].write_text(yaml.safe_dump(config))
    assert run(project, "--no-mlflow") == 0

    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir()) and len(names) == 6
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


@pytest.mark.integration
def test_warnings_do_not_fail_the_evaluation(project, capsys):
    def edit(ck):
        ck["metadata"].update(git_dirty="true", mlflow_run_id="0" * 32)

    edit_checkpoint(project["checkpoint"], edit)
    assert run(project) == 0
    err = capsys.readouterr().err
    assert "uncommitted changes" in err
    assert "MLflow run 00000000000000000000000000000000 was not found" in err


@pytest.mark.integration
def test_clean_test_only(project):
    config = yaml.safe_load(project["config"].read_text()) | {"splits": ["test"]}
    project["config"].write_text(yaml.safe_dump(config))

    assert run(project, "--no-mlflow") == 0

    metrics = json.loads((project["reports"] / "metrics.json").read_text())
    assert "test_official" not in metrics
    assert read_csv(project["reports"] / "excluded_predictions.csv") == []


def test_cli_reports_config_errors(tmp_path, capsys):
    path = tmp_path / "evaluate.yaml"
    path.write_text(yaml.safe_dump(project_eval_config() | {"splits": ["val"]}))
    assert main(["--config", str(path)]) == 1
    assert "error: splits must include test" in capsys.readouterr().err


# --- Pipeline -----------------------------------------------------------------------


def test_dvc_evaluate_stage():
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stage = pipeline["stages"]["evaluate"]

    assert stage["cmd"] == "python -m deepvision.evaluation.evaluate --config configs/evaluate.yaml"
    assert {"artifacts/checkpoints/best.pt", "${split.output}", "src/deepvision/evaluation"} <= set(
        stage["deps"]
    )
    assert {"configs/evaluate.yaml": None} in stage["params"]
    assert stage["metrics"] == [{"reports/evaluation/metrics.json": {"cache": False}}]
    plot = stage["plots"][0]["reports/evaluation/predictions.csv"]
    assert (plot["template"], plot["x"], plot["y"]) == ("confusion", "label", "predicted")
