import csv
import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
import yaml
from mlflow.tracking import MlflowClient
from PIL import Image

from deepvision.training import train
from deepvision.training.engine import FitResult, TrainingError, load_checkpoint
from deepvision.training.train import (
    CliError,
    file_md5,
    flatten,
    git_info,
    main,
    numeric_metrics,
    prepare_checkpoint_dir,
    write_reports,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["a", "b", "c"]


def build_project(root: Path) -> str:
    """Create images, a split manifest, data.yaml, and train.yaml; return the MLflow URI."""
    rng = np.random.default_rng(0)
    rows = []
    for index, label in enumerate(CLASSES):
        for split, count in (("train", 8), ("val", 3)):
            for i in range(count):
                relpath = f"{split}/{label}/{i}.png"
                pixels = rng.integers(0, 120, (32, 32, 3), dtype=np.uint8)
                pixels[..., 0] += np.uint8(60 * index)  # a learnable class signal
                (root / "data" / "images" / relpath).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(root / "data" / "images" / relpath)
                rows.append({"relpath": relpath, "label": label, "split": split})
        # Listed but never written: loading the test split would fail the run.
        rows.append({"relpath": f"test/{label}/missing.png", "label": label, "split": "test"})
    with (root / "data" / "splits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relpath", "label", "split", "group_id"])
        writer.writeheader()
        writer.writerows({**row, "group_id": row["relpath"]} for row in rows)

    (root / "configs").mkdir()
    data = {"archive": {"output": "data/images"}, "split": {"output": "data/splits.csv"}}
    (root / "configs" / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    tracking_uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    config = {
        "seed": 0,
        "data": {
            "config": "configs/data.yaml",
            "class_names": CLASSES,
            "num_classes": 3,
            "image_size": 32,
            "eval_batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "augmentation": {"crop_scale": [0.8, 1.0], "horizontal_flip": 0.5},
        },
        "model": {"backbone": "mobilenet_v3_small", "pretrained": False, "dropout": 0.0},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "mixed_precision": True,
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "scheduler": "cosine",
            "device": "cpu",
            "fine_tune": {
                "head_epochs": 1,
                "head_learning_rate": 0.01,
                "backbone_learning_rate": 0.001,
            },
            "early_stopping": {"monitor": "val_macro_f1", "mode": "max", "patience": 3},
        },
        "output": {"checkpoint_dir": "artifacts/checkpoints", "reports_dir": "reports/training"},
        "mlflow": {"tracking_uri": tracking_uri, "experiment_name": "test-experiment"},
    }
    (root / "configs" / "train.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return tracking_uri


@pytest.fixture
def project(tmp_path, monkeypatch) -> tuple[Path, str]:
    uri = build_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    yield tmp_path, uri
    while mlflow.active_run() is not None:  # never leak a run into other tests
        mlflow.end_run()


def cli(*args: str) -> int:
    return main(["--config", "configs/train.yaml", *args])


def runs(uri: str, experiment: str = "test-experiment") -> list:
    client = MlflowClient(tracking_uri=uri)
    found = client.get_experiment_by_name(experiment)
    return client.search_runs([found.experiment_id]) if found else []


def metric_steps(uri: str, run_id: str, key: str) -> list[int]:
    return [m.step for m in MlflowClient(tracking_uri=uri).get_metric_history(run_id, key)]


def artifacts(uri: str, run_id: str, path: str | None = None) -> set[str]:
    return {a.path for a in MlflowClient(tracking_uri=uri).list_artifacts(run_id, path)}


# --- Full command-line runs ---------------------------------------------------------


@pytest.mark.integration
def test_training_run_writes_outputs_and_logs_to_mlflow(project, capsys):
    root, uri = project

    assert cli() == 0

    out = capsys.readouterr().out
    assert "epoch 1/2 [head]" in out and "epoch 2/2 [full]" in out and "MLflow run:" in out
    for name in ("best.pt", "last.pt"):
        assert (root / "artifacts" / "checkpoints" / name).is_file()
    metrics = json.loads((root / "reports" / "training" / "metrics.json").read_text())
    assert set(metrics) == {
        "best_epoch", "best", "epochs_run", "stopped_early", "completed", "training_seconds",
    }  # fmt: skip
    assert metrics["epochs_run"] == 2 and metrics["completed"] is True
    assert set(metrics["best"]) == set(train.BEST_FIELDS)
    with (root / "reports" / "training" / "history.csv").open(newline="") as f:
        history = list(csv.DictReader(f))
    assert [row["phase"] for row in history] == ["head", "full"]
    assert history[0]["lr_backbone"] == ""

    [run] = runs(uri)
    assert run.info.status == "FINISHED"
    assert run.data.params["training.epochs"] == "2"
    assert run.data.params["model.backbone"] == "mobilenet_v3_small"
    assert run.data.tags["splits_md5"] == file_md5(root / "data" / "splits.csv")
    assert run.data.tags["smoke"] == "false" and run.data.tags["device"] == "cpu"
    assert run.data.tags["mixed_precision"] == "false"  # CPU falls back to full precision
    assert metric_steps(uri, run.info.run_id, "val_macro_f1") == [1, 2]
    assert artifacts(uri, run.info.run_id) == {
        "metrics.json", "history.csv", "train.yaml", "data.yaml",
    }  # fmt: skip
    checkpoint = load_checkpoint(root / "artifacts" / "checkpoints" / "best.pt")
    assert checkpoint["metadata"]["mlflow_run_id"] == run.info.run_id
    assert checkpoint["metadata"]["splits_md5"] == run.data.tags["splits_md5"]


@pytest.mark.integration
def test_resume_continues_the_same_mlflow_run(project, capsys):
    root, uri = project

    assert cli("--max-epochs", "1") == 0
    assert not json.loads((root / "reports" / "training" / "metrics.json").read_text())["completed"]
    assert cli() == 1  # refuses to overwrite the unfinished run
    assert "--resume" in capsys.readouterr().err
    assert cli("--resume") == 0

    [run] = runs(uri)
    assert run.info.status == "FINISHED"
    assert run.data.tags["resumed"] == "true"
    assert metric_steps(uri, run.info.run_id, "val_macro_f1") == [1, 2]
    metrics = json.loads((root / "reports" / "training" / "metrics.json").read_text())
    assert (metrics["epochs_run"], metrics["completed"]) == (2, True)


@pytest.mark.integration
def test_restart_replaces_only_the_checkpoints(project):
    root, uri = project
    checkpoints = root / "artifacts" / "checkpoints"
    checkpoints.mkdir(parents=True)
    for name in ("best.pt", "last.pt", "keep.txt"):
        (checkpoints / name).write_text("old")

    assert cli("--restart", "--no-mlflow") == 0

    assert (checkpoints / "keep.txt").read_text() == "old"
    assert load_checkpoint(checkpoints / "best.pt")["epoch"] in (1, 2)
    assert not (root / "mlflow.db").exists() and not (root / "mlruns").exists()


@pytest.mark.integration
def test_smoke_run_is_isolated_and_can_log_the_checkpoint(project):
    root, uri = project

    assert cli("--smoke", "--log-checkpoint") == 0

    assert not (root / "artifacts").exists() and not (root / "reports").exists()
    assert (root / "tmp" / "smoke" / "checkpoints" / "best.pt").is_file()
    history = (root / "tmp" / "smoke" / "reports" / "history.csv").read_text().splitlines()
    assert len(history) == 3  # header + one head epoch + one full epoch
    assert runs(uri) == []
    [run] = runs(uri, "deepvision-smoke")
    assert (run.data.tags["smoke"], run.data.tags["max_batches"]) == ("true", "2")
    assert artifacts(uri, run.info.run_id, "checkpoints") == {"checkpoints/best.pt"}

    assert cli("--smoke", "--max-batches", "1") == 0  # starts fresh; explicit limit wins
    tags = [r.data.tags["max_batches"] for r in runs(uri, "deepvision-smoke")]
    assert sorted(tags) == ["1", "2"]


@pytest.mark.integration
def test_resuming_an_untracked_run_starts_a_complete_mlflow_run(project):
    _, uri = project

    assert cli("--no-mlflow", "--max-epochs", "1") == 0
    assert cli("--resume") == 0

    [run] = runs(uri)
    assert run.info.status == "FINISHED"
    assert run.data.params["training.epochs"] == "2"  # parameters are logged for the new run
    assert metric_steps(uri, run.info.run_id, "val_macro_f1") == [2]


@pytest.mark.integration
def test_training_errors_mark_the_run_failed(project, monkeypatch, capsys):
    _, uri = project

    def broken_fit(*args, **kwargs):
        raise TrainingError("non-finite training loss (nan) in epoch 1, batch 3")

    monkeypatch.setattr(train, "fit", broken_fit)
    assert cli() == 1
    assert "error: non-finite training loss" in capsys.readouterr().err
    assert runs(uri)[0].info.status == "FAILED"


@pytest.mark.integration
def test_interruption_marks_the_run_killed(project, monkeypatch, capsys):
    _, uri = project

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "fit", interrupted)
    assert cli() == 130
    assert "--resume" in capsys.readouterr().err
    assert runs(uri)[0].info.status == "KILLED"


# --- Argument and state checks (no training) ----------------------------------------


def test_existing_state_requires_resume_or_restart(tmp_path):
    (tmp_path / "last.pt").write_text("x")
    with pytest.raises(CliError, match="--resume to continue it or --restart"):
        prepare_checkpoint_dir(tmp_path, resume=False, restart=False)
    assert prepare_checkpoint_dir(tmp_path, resume=True, restart=False) == tmp_path / "last.pt"

    (tmp_path / "best.pt").write_text("x")
    assert prepare_checkpoint_dir(tmp_path, resume=False, restart=True) is None
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(CliError, match="does not exist"):
        prepare_checkpoint_dir(tmp_path, resume=True, restart=False)
    assert prepare_checkpoint_dir(tmp_path / "new", resume=False, restart=False) is None


def test_invalid_flag_combinations(project, capsys):
    assert cli("--smoke", "--resume") == 1
    assert "--smoke cannot be combined" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli("--resume", "--restart")
    assert cli("--resume") == 1
    assert "does not exist" in capsys.readouterr().err
    for flag, value in (("--max-epochs", "0"), ("--max-batches", "-1"), ("--max-epochs", "x")):
        with pytest.raises(SystemExit):
            cli(flag, value)


def test_invalid_config_is_reported(project, capsys):
    root, _ = project
    config = yaml.safe_load((root / "configs" / "train.yaml").read_text())
    config["training"]["epochs"] = 0
    (root / "configs" / "train.yaml").write_text(yaml.safe_dump(config))

    assert cli() == 1
    assert "error: training.epochs" in capsys.readouterr().err


# --- Helpers ----------------------------------------------------------------------


def test_reports_are_written_from_the_best_epoch(tmp_path):
    def record(epoch, f1, peak=None):
        values = {
            "epoch": epoch, "phase": "full", "train_loss": 1.0, "train_accuracy": 0.5,
            "val_loss": 2.0 - f1, "val_accuracy": f1, "val_macro_precision": f1,
            "val_macro_recall": f1, "val_macro_f1": f1, "lr_head": 0.01, "lr_backbone": None,
            "epoch_seconds": 1.5, "improved": True,
        }  # fmt: skip
        return values | ({"peak_gpu_memory_mib": peak} if peak else {})

    result = FitResult([record(1, 0.5, 300), record(2, 0.8, 420)], 2, 0.8, False, True, None, None)
    metrics = write_reports(tmp_path, result)

    assert metrics["best_epoch"] == 2 and metrics["best"]["val_macro_f1"] == 0.8
    assert (metrics["epochs_run"], metrics["training_seconds"]) == (2, 3.0)
    assert metrics["peak_gpu_memory_mib"] == 420
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    with (tmp_path / "history.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == train.HISTORY_FIELDS
    assert rows[0]["lr_backbone"] == "" and rows[1]["peak_gpu_memory_mib"] == "420"


def test_metric_and_parameter_helpers():
    record = {"epoch": 3, "phase": "full", "val_loss": 0.5, "lr_backbone": None, "improved": True}
    assert numeric_metrics(record) == {"val_loss": 0.5, "improved": 1.0}
    assert flatten({"a": {"b": 1, "c": {"d": [1, 2]}}, "e": None}) == {
        "a.b": "1",
        "a.c.d": "[1, 2]",
        "e": "None",
    }


def test_file_md5_and_git_info(tmp_path, monkeypatch):
    path = tmp_path / "data.bin"
    path.write_bytes(b"deepvision" * 1000)
    assert file_md5(path) == hashlib.md5(path.read_bytes()).hexdigest()

    monkeypatch.chdir(REPO_ROOT)
    info = git_info()
    assert len(info["git_commit"]) == 40 and info["git_dirty"] in ("true", "false")
    monkeypatch.chdir(tmp_path)  # not inside a Git repository
    assert git_info() == {"git_commit": "unknown", "git_dirty": "unknown"}


# --- Pipeline configuration -------------------------------------------------------


def test_dvc_train_stage_and_plots():
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stage = pipeline["stages"]["train"]

    assert stage["cmd"].endswith("--config configs/train.yaml --restart")
    assert {"${split.output}", "${archive.output}", "src/deepvision/training"} <= set(stage["deps"])
    assert stage["params"] == [{"configs/train.yaml": ["seed", "data", "model", "training"]}]
    assert stage["outs"] == ["artifacts/checkpoints/best.pt"]
    assert stage["metrics"] == [{"reports/training/metrics.json": {"cache": False}}]
    plotted = [next(iter(plot.values()))["y"] for plot in pipeline["plots"]]
    assert {"reports/training/history.csv": ["train_loss", "val_loss"]} in plotted


def test_python_bytecode_is_ignored_by_dvc(monkeypatch):
    from dvc.repo import Repo

    monkeypatch.chdir(REPO_ROOT)
    ignore = Repo(".").dvcignore
    assert ignore.is_ignored_file("src/deepvision/training/__pycache__/engine.cpython-312.pyc")
    assert not ignore.is_ignored_file("src/deepvision/training/engine.py")


def test_training_never_requests_the_test_split(project, monkeypatch):
    requested = []
    real = train.build_dataloaders

    def spy(config, splits, **kwargs):
        requested.append(tuple(splits))
        return real(config, splits, **kwargs)

    monkeypatch.setattr(train, "build_dataloaders", spy)
    monkeypatch.setattr(train, "fit", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    assert cli("--no-mlflow") == 130
    assert requested == [("train", "val")]
