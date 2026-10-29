import contextlib
import csv
import json
import socket
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import CLASSES, make_checkpoint
from PIL import Image

from deepvision.monitoring import drift
from deepvision.monitoring.drift import (
    DriftError,
    image_features,
    load_drift_config,
    main,
    split_relpaths,
    summarize_drift,
)
from deepvision.training.config import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]
COLUMNS = len(drift.IMAGE_FEATURES) + 2 + len(CLASSES) + 1  # + confidence, entropy, label


def project_drift_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "drift.yaml").read_text(encoding="utf-8"))


# --- Configuration ----------------------------------------------------------------------


def test_project_drift_config_is_valid():
    config = load_drift_config(REPO_ROOT / "configs" / "drift.yaml")
    assert (config.reference_split, config.current_split) == ("val", "pred")
    assert config.checkpoint == Path("artifacts/checkpoints/best.pt")
    assert config.drift_share == 0.5


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"current_split": "test"}, "held-out test splits are not used"),
        ({"current_split": "test_official"}, "held-out test splits are not used"),
        ({"reference_split": "test_excluded"}, "held-out test splits are not used"),
        ({"current_split": "val"}, "must differ"),
        ({"batch_size": 0}, "batch_size"),
        ({"device": "tpu"}, "device"),
        ({"drift_share": 0}, "drift_share"),
        ({"drift_share": 1.5}, "drift_share"),
        ({"drift_share": "half"}, "drift_share must be a number"),
        ({"drift_share": True}, "drift_share must be a number"),
        ({"extra": 1}, "unknown \\['extra'\\]"),
    ],
)
def test_invalid_drift_config_is_rejected(tmp_path, change, message):
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(project_drift_config() | change))
    with pytest.raises(ConfigError, match=message):
        load_drift_config(path)


def test_test_splits_cannot_be_read_directly(tmp_path):
    with pytest.raises(DriftError, match="not allowed"):
        split_relpaths(tmp_path / "unused.csv", "test")


# --- Image features -----------------------------------------------------------------------


def solid(color, size=(20, 10)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_features_of_solid_images():
    black, white, red = (image_features(solid(c)) for c in ("black", "white", (255, 0, 0)))
    assert black["brightness"] == 0 and white["brightness"] == pytest.approx(1, abs=1e-6)
    assert black["contrast"] == 0 and black["entropy"] == 0 and black["sharpness"] == 0
    assert red["red_mean"] == 1 and red["green_mean"] == 0 and red["saturation"] == 1
    assert white["saturation"] == 0 and black["saturation"] == 0  # no division by zero
    assert (black["width"], black["height"]) == (20, 10)


def test_sharpness_contrast_and_entropy_respond_to_structure():
    checker = np.indices((32, 32)).sum(axis=0) % 2 * 255
    sharp = Image.fromarray(np.repeat(checker[..., None], 3, axis=2).astype(np.uint8))
    halves = np.zeros((32, 32, 3), dtype=np.uint8)
    halves[:, 16:] = 255
    features = image_features(sharp)
    assert features["sharpness"] > image_features(Image.fromarray(halves))["sharpness"] > 0
    assert features["contrast"] == pytest.approx(0.5, abs=1e-6)
    assert features["entropy"] == pytest.approx(1.0)  # two equally frequent levels: 1 bit


def test_tiny_images_have_zero_sharpness():
    assert image_features(solid("gray", (2, 2)))["sharpness"] == 0


# --- Summary of Evidently results ------------------------------------------------------


def value_drift(column, method, threshold, value) -> dict:
    return {
        "config": {"type": "evidently:metric_v2:ValueDrift", "column": column, "method": method,
                   "threshold": threshold},
        "value": value,
    }  # fmt: skip


def snapshot(count, *drifts) -> dict:
    total = {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
             "value": {"count": count, "share": count / len(drifts)}}  # fmt: skip
    return {"metrics": [total, *drifts]}


GROUPS = {"input": ["a", "b"], "prediction": ["c", "d"]}


def test_summary_applies_p_value_and_distance_rules():
    result = summarize_drift(
        snapshot(
            2,
            value_drift("a", "K-S p_value", 0.05, 0.01),  # p below threshold: drift
            value_drift("b", "K-S p_value", 0.05, 0.40),
            value_drift("c", "Wasserstein distance (normed)", 0.1, 0.10),  # at threshold: drift
            value_drift("d", "Jensen-Shannon distance", 0.1, 0.02),
        ),
        drift_share=0.5,
        groups=GROUPS,
    )
    assert result["drifted_column_names"] == ["a", "c"]
    assert result["drifted_share"] == 0.5 and result["dataset_drift"] is True
    assert result["columns"]["c"] == {
        "method": "Wasserstein distance (normed)", "threshold": 0.1, "score": 0.1, "drifted": True,
    }  # fmt: skip
    assert result["groups"]["input"] == {
        "drift": True, "drifted_columns": 1, "columns_tested": 2, "drifted_share": 0.5,
    }  # fmt: skip


def test_one_drifting_group_is_dataset_drift():
    # Input statistics all drift, model outputs do not: 2/4 overall, but input drift.
    drifts = (
        value_drift("a", "K-S p_value", 0.05, 0.001),
        value_drift("b", "K-S p_value", 0.05, 0.001),
        value_drift("c", "K-S p_value", 0.05, 0.9),
        value_drift("d", "Jensen-Shannon distance", 0.1, 0.01),
    )
    result = summarize_drift(snapshot(2, *drifts), drift_share=0.75, groups=GROUPS)
    assert result["drifted_share"] == 0.5  # below 0.75 overall
    assert result["groups"]["input"]["drift"] is True
    assert result["groups"]["prediction"]["drift"] is False
    assert result["dataset_drift"] is True


def test_no_group_reaching_the_share_is_no_drift():
    drifts = (
        value_drift("a", "K-S p_value", 0.05, 0.001),
        value_drift("b", "K-S p_value", 0.05, 0.9),
        value_drift("c", "K-S p_value", 0.05, 0.001),
        value_drift("d", "Jensen-Shannon distance", 0.1, 0.01),
    )
    result = summarize_drift(snapshot(2, *drifts), drift_share=0.75, groups=GROUPS)
    assert result["dataset_drift"] is False


def test_summary_must_agree_with_evidently():
    one = value_drift("a", "K-S p_value", 0.05, 0.01)
    with pytest.raises(DriftError, match="Evidently counted 0"):
        summarize_drift(snapshot(0, one), 0.5, {"input": ["a"]})
    with pytest.raises(DriftError, match="no column drift results"):
        summarize_drift({"metrics": []}, 0.5, GROUPS)
    with pytest.raises(DriftError, match="do not match"):
        summarize_drift(snapshot(1, one), 0.5, {"input": ["a", "missing"]})


# --- End to end -------------------------------------------------------------------------


def build_project(root: Path) -> None:
    """val and pred images from the same distribution, plus test images that must stay unread."""
    rng = np.random.default_rng(0)
    rows = []
    for split, count in (("val", 90), ("pred", 90), ("test", 5)):
        for i in range(count):
            base = rng.uniform(40, 200)
            pixels = np.clip(rng.normal(base, 30, (32, 32, 3)), 0, 255).astype(np.uint8)
            relpath = f"{split}/{i}.png"
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(root / "images" / relpath)
            rows.append({"relpath": relpath, "split": split, "label": "" if split == "pred" else
                         CLASSES[i % len(CLASSES)]})  # fmt: skip
    with (root / "splits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relpath", "split", "label"])
        writer.writeheader()
        writer.writerows(rows)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"archive": {"output": "images"}, "split": {"output": "splits.csv"}})
    )
    train = {
        "seed": 0,
        "data": {"config": "data.yaml", "class_names": CLASSES, "num_classes": len(CLASSES),
                 "image_size": 32, "eval_batch_size": 8, "num_workers": 0, "pin_memory": False},
        "training": {"batch_size": 8},
    }  # fmt: skip
    (root / "train.yaml").write_text(yaml.safe_dump(train))
    make_checkpoint(root / "best.pt", image_size=32)
    config = project_drift_config() | {
        "checkpoint": "best.pt", "train_config": "train.yaml", "batch_size": 16, "device": "cpu",
        "features_dir": "out/features", "reports_dir": "out/reports",
    }  # fmt: skip
    (root / "drift.yaml").write_text(yaml.safe_dump(config))


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("drift")
    build_project(root)
    return root


@pytest.fixture
def opened(monkeypatch) -> list[str]:
    """Records every image path the drift pipeline loads."""
    paths, real = [], drift.load_image

    def spy(path):
        paths.append(Path(path).as_posix())
        return real(path)

    monkeypatch.setattr(drift, "load_image", spy)
    return paths


@pytest.fixture
def no_network(monkeypatch):
    """Fails on any connection that leaves the machine. Loopback stays allowed: importing
    Evidently starts an asyncio loop, whose wake-up socket pair is a loopback TCP
    connection on Windows."""
    real_connect, real_create = socket.socket.connect, socket.create_connection

    def check(address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"drift analysis must not connect to {address!r}")

    def connect(self, address):
        check(address)
        return real_connect(self, address)

    def create_connection(address, *args, **kwargs):
        check(address)
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", create_connection)


def test_same_distribution_shows_no_dataset_drift(project, opened, no_network, capsys):
    with contextlib.chdir(project):
        assert main(["--config", "drift.yaml", "--fail-on-drift"]) == 0

    summary = json.loads((project / "out/reports/summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_drift"] is False
    assert summary["reference"] == {"split": "val", "images": 90}
    assert summary["current"] == {"split": "pred", "images": 90}
    assert summary["columns_tested"] == COLUMNS and summary["simulated"] is None
    assert summary["groups"]["input"]["columns_tested"] == len(drift.IMAGE_FEATURES)
    assert summary["groups"]["prediction"]["columns_tested"] == len(CLASSES) + 3
    assert not summary["groups"]["input"]["drift"] and not summary["groups"]["prediction"]["drift"]
    assert summary["model_version"].startswith("mobilenet_v3_small-")
    assert set(summary["predicted_label_share"]) == set(CLASSES)
    assert "brightness" not in summary["drifted_column_names"]
    assert "no dataset drift" in capsys.readouterr().out

    with (project / "out/features/current.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 90 and rows[0]["relpath"] == "pred/0.png"
    assert {"brightness", "predicted_label", "prediction_entropy", "prob_sea"} <= set(rows[0])
    html = (project / "out/reports/report.html").read_text(encoding="utf-8")
    assert len(html) > 100_000 and "<script" in html
    # Only the configured splits were read: never the held-out test images.
    assert len(opened) == 180 and not any("/images/test/" in p for p in opened)


def test_simulated_darkening_is_detected(project, tmp_path, no_network, capsys):
    with contextlib.chdir(project):
        code = main(["--config", "drift.yaml", "--simulate", "darken", "--output-dir",
                     str(tmp_path / "sim")])  # fmt: skip
    assert code == 0
    summary = json.loads((tmp_path / "sim/summary.json").read_text(encoding="utf-8"))
    assert summary["simulated"] == "darken"
    assert {"brightness", "red_mean", "green_mean", "blue_mean"} <= set(
        summary["drifted_column_names"]
    )
    # Darkening is an input change: it must be reported even if predictions barely move.
    assert summary["groups"]["input"]["drift"] is True and summary["dataset_drift"] is True
    means = summary["feature_means"]["brightness"]
    assert means["current"] < 0.7 * means["reference"]
    assert (tmp_path / "sim/features/current.csv").is_file()
    assert (tmp_path / "sim/report.html").is_file()


def test_fail_on_drift_exit_code(project, monkeypatch, capsys):
    def drifting(config, simulate=None):
        return {"dataset_drift": True, "drifted_columns": 12, "columns_tested": 19,
                "drifted_share": 0.63, "drift_share_threshold": 0.5,
                "groups": {"input": {"drift": True, "drifted_columns": 8, "columns_tested": 10}},
                "drifted_column_names": [], "columns": {}, "simulated": None,
                "reference": {"split": "val", "images": 1},
                "current": {"split": "pred", "images": 1}}  # fmt: skip

    monkeypatch.setattr(drift, "run_drift", drifting)
    with contextlib.chdir(project):
        assert main(["--config", "drift.yaml"]) == 0
        assert main(["--config", "drift.yaml", "--fail-on-drift"]) == drift.DRIFT_EXIT_CODE
    assert "DRIFT DETECTED" in capsys.readouterr().out


def test_simulate_requires_output_dir(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--simulate", "blur"])
    assert exit_info.value.code == 2
    assert "requires --output-dir" in capsys.readouterr().err


def test_missing_checkpoint_is_a_clean_error(project, tmp_path, capsys):
    config = yaml.safe_load((project / "drift.yaml").read_text()) | {"checkpoint": "missing.pt"}
    (tmp_path / "drift.yaml").write_text(yaml.safe_dump(config))
    with contextlib.chdir(project):
        assert main(["--config", str(tmp_path / "drift.yaml")]) == 1
    assert "error: checkpoint not found" in capsys.readouterr().err


def test_telemetry_is_disabled(monkeypatch):
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("EVIDENTLY_DISABLE_TELEMETRY", raising=False)
    drift._disable_telemetry()
    assert drift.os.environ["DO_NOT_TRACK"] == "1"
    assert drift.os.environ["EVIDENTLY_DISABLE_TELEMETRY"] == "1"


# --- DVC stage ----------------------------------------------------------------------------


def test_dvc_drift_stage():
    stage = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]["drift"]
    assert stage["cmd"] == "python -m deepvision.monitoring.drift --config configs/drift.yaml"
    required = {
        "artifacts/checkpoints/best.pt", "${split.output}", "${archive.output}",
        "src/deepvision/monitoring", "src/deepvision/inference",
    }  # fmt: skip
    assert required <= set(stage["deps"])
    assert stage["metrics"] == [{"reports/drift/summary.json": {"cache": False}}]
    outs = {next(iter(o)) if isinstance(o, dict) else o for o in stage["outs"]}
    assert outs == {
        "reports/drift/report.html", "data/processed/drift/reference.csv",
        "data/processed/drift/current.csv",
    }  # fmt: skip
