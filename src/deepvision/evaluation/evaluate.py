"""Held-out test evaluation of a trained checkpoint.

    python -m deepvision.evaluation.evaluate --config configs/evaluate.yaml

The test set measures the finished model; it is never used to choose between models
(see docs/evaluation.md). The model is rebuilt from the checkpoint's own metadata
(backbone, classes, image size, normalization) with pretrained=False, so nothing is
downloaded and later config edits cannot change what is measured. Inference runs in
full precision.

Before any test image is loaded, the validation split is re-scored and compared with
the validation metrics stored in the checkpoint; a mismatch beyond the configured
tolerance (training validated with FP16 on CUDA) stops the run.

Reports (in reports_dir):
    metrics.json                 headline metrics with 95% bootstrap intervals (DVC metrics)
    per_class.csv                precision, recall, F1, support per class and split
    confusion_matrix.csv         clean test counts (rows = true class)
    predictions.csv              one row per clean test image (DVC confusion plot)
    excluded_predictions.csv     the test_excluded images (label conflicts), if evaluated
    errors.csv                   the most confident clean test errors, for review

Test metrics are also logged to the checkpoint's MLflow training run as test_* and
test_official_*. Exit codes: 0 success, 1 error.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import yaml

from deepvision.data.dataset import (
    DataConfig,
    DatasetImageError,
    ManifestError,
    build_dataloaders,
    load_manifest,
)
from deepvision.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from deepvision.evaluation.metrics import summarize
from deepvision.models.factory import ImageClassifier, build_model
from deepvision.training.config import ConfigError, load_train_config
from deepvision.training.engine import (
    CHECKPOINT_VERSION,
    classification_metrics,
    confusion_matrix,
    load_checkpoint,
    resolve_device,
)
from deepvision.training.train import file_md5, git_info

EVALUATION_SPLITS = ("test", "test_official")
DEVICES = frozenset({"auto", "cpu", "cuda"})
CONFIG_KEYS = {
    "checkpoint", "train_config", "reports_dir", "splits", "batch_size", "device",
    "validation_check_tolerance", "bootstrap", "top_errors",
}  # fmt: skip
CHECKED_METRICS = ("accuracy", "macro_f1")


class EvaluationError(RuntimeError):
    """Raised when the checkpoint cannot be evaluated faithfully."""


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint: Path
    train_config: Path
    reports_dir: Path
    splits: tuple[str, ...]
    batch_size: int
    device: str
    validation_check_tolerance: float
    bootstrap_samples: int
    bootstrap_seed: int
    top_errors: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def load_evaluation_config(path: Path = Path("configs/evaluate.yaml")) -> EvaluationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), f"{path} must contain a mapping")
    unknown, missing = set(raw) - CONFIG_KEYS, CONFIG_KEYS - set(raw)
    _require(
        not unknown and not missing, f"{path}: unknown {sorted(unknown)}, missing {sorted(missing)}"
    )
    try:
        config = EvaluationConfig(
            checkpoint=Path(raw["checkpoint"]),
            train_config=Path(raw["train_config"]),
            reports_dir=Path(raw["reports_dir"]),
            splits=tuple(raw["splits"]),
            batch_size=raw["batch_size"],
            device=raw["device"],
            validation_check_tolerance=raw["validation_check_tolerance"],
            bootstrap_samples=raw["bootstrap"]["samples"],
            bootstrap_seed=raw["bootstrap"]["seed"],
            top_errors=raw["top_errors"],
        )
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"invalid evaluation configuration {path}: {exc!r}") from exc
    _require("test" in config.splits, "splits must include test, the primary clean test set")
    _require(
        set(config.splits) <= set(EVALUATION_SPLITS)
        and len(set(config.splits)) == len(config.splits),
        f"splits must be distinct values from {list(EVALUATION_SPLITS)}",
    )
    _require(config.batch_size >= 1, "batch_size must be at least 1")
    _require(config.device in DEVICES, f"device must be one of {sorted(DEVICES)}")
    _require(config.validation_check_tolerance >= 0, "validation_check_tolerance must be >= 0")
    _require(config.bootstrap_samples >= 0, "bootstrap.samples must be >= 0 (0 disables it)")
    _require(config.top_errors >= 0, "top_errors must be >= 0")
    return config


# --- Model --------------------------------------------------------------------------


def load_model(
    checkpoint_path: Path, class_names: tuple[str, ...], device: torch.device
) -> tuple[ImageClassifier, dict]:
    """Rebuild the trained model from its checkpoint, checking it matches the project."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise EvaluationError(f"checkpoint not found: {checkpoint_path}; run `dvc repro train`")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION:
        raise EvaluationError(f"unsupported checkpoint format {checkpoint.get('format_version')}")
    if checkpoint["class_names"] != list(class_names):
        raise EvaluationError(
            f"checkpoint classes {checkpoint['class_names']} differ from the configured "
            f"class_names {list(class_names)}"
        )
    expected = {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
    if checkpoint["normalization"] != expected:
        raise EvaluationError(
            f"checkpoint normalization {checkpoint['normalization']} is not ImageNet's"
        )
    model = build_model(checkpoint["backbone"], checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def predict(
    model: ImageClassifier, loader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-precision logits and targets, in loader order."""
    model.eval()
    logits, targets = [], []
    for images, labels in loader:
        logits.append(model(images.to(device, non_blocking=True)).float().cpu())
        targets.append(labels)
    return torch.cat(logits), torch.cat(targets)


def check_validation(
    model: ImageClassifier,
    data: DataConfig,
    checkpoint: dict,
    device: torch.device,
    tolerance: float,
) -> dict:
    """Re-score validation and compare with the checkpoint's stored validation metrics."""
    loader = build_dataloaders(data, ("val",), device=device)["val"]
    logits, targets = predict(model, loader, device)
    recomputed = classification_metrics(
        confusion_matrix(targets, logits.argmax(dim=1), len(data.class_names))
    )
    stored = checkpoint["metrics"]
    result = {"tolerance": tolerance}
    for key in CHECKED_METRICS:
        result[f"stored_{key}"] = stored[key]
        result[f"recomputed_{key}"] = recomputed[key]
    result["max_difference"] = max(abs(recomputed[k] - stored[k]) for k in CHECKED_METRICS)
    if result["max_difference"] > tolerance:
        raise EvaluationError(
            f"validation metrics recomputed from the checkpoint differ from the stored ones by "
            f"{result['max_difference']:.4f} (> {tolerance}); preprocessing or weights do not "
            "match training, so the test set was not evaluated"
        )
    return result


# --- Reports ------------------------------------------------------------------------


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prediction_rows(
    paths: list[str], probabilities: torch.Tensor, targets: torch.Tensor, class_names
) -> list[dict]:
    rows = []
    for path, probs, target in zip(paths, probabilities, targets, strict=True):
        predicted = int(probs.argmax())
        rows.append(
            {
                "relpath": path,
                "label": class_names[int(target)],
                "predicted": class_names[predicted],
                "confidence": f"{probs[predicted].item():.6f}",
                "correct": predicted == int(target),
                **{f"prob_{name}": f"{probs[i].item():.6f}" for i, name in enumerate(class_names)},
            }
        )
    return rows


def write_reports(
    reports_dir: Path,
    metrics: dict,
    results: dict,
    clean_rows: list[dict],
    excluded_rows: list[dict],
    class_names: list[str],
    top_errors: int,
) -> list[Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    prediction_fields = ["relpath", "label", "predicted", "confidence", "correct"]
    prediction_fields += [f"prob_{name}" for name in class_names]

    paths = {
        name: reports_dir / name
        for name in (
            "metrics.json", "per_class.csv", "confusion_matrix.csv", "predictions.csv",
            "excluded_predictions.csv", "errors.csv",
        )
    }  # fmt: skip
    paths["metrics.json"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(
        paths["per_class.csv"],
        ["split", "class", "precision", "recall", "f1", "support"],
        [
            {"split": split, **row}
            for split, result in results.items()
            for row in result["per_class"]
        ],
    )
    matrix = results["test"]["confusion_matrix"].tolist()
    _write_csv(
        paths["confusion_matrix.csv"],
        ["label", *class_names],
        [
            {"label": name, **dict(zip(class_names, counts, strict=True))}
            for name, counts in zip(class_names, matrix, strict=True)
        ],
    )
    _write_csv(paths["predictions.csv"], prediction_fields, clean_rows)
    _write_csv(paths["excluded_predictions.csv"], prediction_fields, excluded_rows)
    errors = sorted(
        (row for row in clean_rows if not row["correct"]),
        key=lambda row: (-float(row["confidence"]), row["relpath"]),
    )[:top_errors]
    _write_csv(
        paths["errors.csv"],
        ["relpath", "label", "predicted", "confidence", "label_probability"],
        [
            {
                "relpath": row["relpath"],
                "label": row["label"],
                "predicted": row["predicted"],
                "confidence": row["confidence"],
                "label_probability": row[f"prob_{row['label']}"],
            }
            for row in errors
        ],
    )
    return list(paths.values())


# --- MLflow -------------------------------------------------------------------------


def mlflow_metrics(metrics: dict) -> dict[str, float]:
    """Flatten split headlines to test_* / test_official_* MLflow metric names."""
    flat = {}
    for split in EVALUATION_SPLITS:
        for key, value in metrics.get(split, {}).items():
            if isinstance(value, dict):
                for bound, number in value.items():
                    flat[f"{split}_{key}_{bound}"] = float(number)
            else:
                flat[f"{split}_{key}"] = float(value)
    return flat


def log_to_mlflow(
    tracking_uri: str, run_id: str, metrics: dict, artifacts: list[Path]
) -> str | None:
    """Add test metrics and reports to the training run; return a warning instead of failing."""
    if not run_id:
        return "the checkpoint has no MLflow run id; test metrics were not logged to MLflow"
    from mlflow.entities import Metric
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri)
    try:
        client.get_run(run_id)
    except MlflowException as exc:
        return f"MLflow run {run_id} was not found ({exc.error_code}); test metrics not logged"
    timestamp = int(time.time() * 1000)
    entries = [Metric(k, v, timestamp, 0) for k, v in sorted(mlflow_metrics(metrics).items())]
    client.log_batch(run_id, metrics=entries)
    client.set_tag(run_id, "evaluated", "true")
    client.set_tag(run_id, "evaluation_commit", metrics["evaluation_commit"])
    for path in artifacts:
        client.log_artifact(run_id, str(path), artifact_path="evaluation")
    return None


# --- Main ---------------------------------------------------------------------------


def run_evaluation(config: EvaluationConfig, *, use_mlflow: bool = True) -> dict:
    train_config = load_train_config(config.train_config)
    class_names = list(train_config.data.class_names)
    device = resolve_device(config.device)
    model, checkpoint = load_model(config.checkpoint, train_config.data.class_names, device)
    if checkpoint["metadata"].get("git_dirty") == "true":
        print(
            "warning: the checkpoint was trained from a tree with uncommitted changes",
            file=sys.stderr,
        )
    data = replace(
        train_config.data, image_size=checkpoint["image_size"], eval_batch_size=config.batch_size
    )

    validation = check_validation(
        model, data, checkpoint, device, config.validation_check_tolerance
    )

    # One pass over test_official (= test + test_excluded) when requested, else test only.
    split = "test_official" if "test_official" in config.splits else "test"
    loader = build_dataloaders(data, (split,), device=device)[split]
    logits, targets = predict(model, loader, device)
    probabilities = torch.softmax(logits.double(), dim=1)
    source = {r["relpath"]: r["split"] for r in load_manifest(data.manifest, class_names)}
    paths = [path for path, _ in loader.dataset.samples]
    clean = torch.tensor([source[path] == "test" for path in paths])

    def score(mask: torch.Tensor) -> dict:
        return summarize(
            probabilities[mask],
            targets[mask],
            class_names,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.bootstrap_seed,
        )

    results = {"test": score(clean)}
    if split == "test_official":
        results["test_official"] = score(torch.ones_like(clean))
    rows = prediction_rows(paths, probabilities, targets, class_names)
    clean_rows = [row for row, keep in zip(rows, clean.tolist(), strict=True) if keep]
    excluded_rows = [row for row, keep in zip(rows, clean.tolist(), strict=True) if not keep]

    metadata = checkpoint["metadata"]
    metrics = {
        "checkpoint": {
            "path": Path(config.checkpoint).as_posix(),
            "md5": file_md5(config.checkpoint),
            "epoch": checkpoint["epoch"],
            "backbone": checkpoint["backbone"],
            "git_commit": metadata.get("git_commit", "unknown"),
            "mlflow_run_id": metadata.get("mlflow_run_id", ""),
        },
        "evaluation_commit": git_info()["git_commit"],
        "validation_check": validation,
        **{name: result["headline"] for name, result in results.items()},
    }
    reports = write_reports(
        config.reports_dir,
        metrics,
        results,
        clean_rows,
        excluded_rows,
        class_names,
        config.top_errors,
    )
    if use_mlflow:
        warning = log_to_mlflow(
            train_config.mlflow.tracking_uri, metadata.get("mlflow_run_id", ""), metrics, reports
        )
        if warning:
            print(f"warning: {warning}", file=sys.stderr)
    return metrics


def format_summary(metrics: dict) -> str:
    check = metrics["validation_check"]
    lines = [
        f"validation check passed: macro-F1 {check['recomputed_macro_f1']:.4f} "
        f"(stored {check['stored_macro_f1']:.4f}, difference {check['max_difference']:.4f})"
    ]
    for split in EVALUATION_SPLITS:
        if split not in metrics:
            continue
        m = metrics[split]
        interval = m.get("macro_f1_ci95")
        bounds = f" [95% CI {interval['low']:.4f}-{interval['high']:.4f}]" if interval else ""
        lines.append(
            f"{split} ({m['images']} images): macro-F1 {m['macro_f1']:.4f}{bounds} | "
            f"accuracy {m['accuracy']:.4f} | top-2 {m['top2_accuracy']:.4f} | "
            f"log loss {m['log_loss']:.4f} | ECE {m['ece']:.4f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepvision.evaluation.evaluate",
        description="Evaluate the trained checkpoint on the held-out test set.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate.yaml"))
    parser.add_argument("--no-mlflow", action="store_true", help="do not log to MLflow")
    parser.add_argument("--device", choices=sorted(DEVICES), help="override the configured device")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_evaluation_config(args.config)
        if args.device:
            config = replace(config, device=args.device)
        metrics = run_evaluation(config, use_mlflow=not args.no_mlflow)
    except (EvaluationError, ConfigError, ManifestError, DatasetImageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_summary(metrics))
    print(f"reports written to {config.reports_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
