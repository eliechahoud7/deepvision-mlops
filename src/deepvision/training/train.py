"""Command-line entry point for training, with MLflow tracking and DVC-friendly reports.

    python -m deepvision.training.train --config configs/train.yaml [--resume | --restart]

Only the train and validation splits are loaded; the test set is never touched here.
Existing training state (best.pt or last.pt) is never overwritten silently: pass
--resume to continue from last.pt or --restart to delete both files and start over.
The DVC train stage runs with --restart, because DVC always reruns a stage from scratch.

--smoke runs two short epochs (one per fine-tuning phase) on two batches per split,
writes only under tmp/smoke/, and logs to a separate MLflow experiment.

Outputs:
    <checkpoint_dir>/best.pt, last.pt      model checkpoints (see engine.py)
    <reports_dir>/metrics.json             best-epoch validation metrics (DVC metrics)
    <reports_dir>/history.csv              one row per epoch (DVC plots)

Exit codes: 0 success, 1 configuration/data/training error, 130 interrupted.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # quiet MLflow's import-time hint

import torch
import torchvision
import yaml

from deepvision.data.dataset import DatasetImageError, ManifestError, build_dataloaders
from deepvision.models.factory import build_model
from deepvision.training.config import ConfigError, TrainConfig, load_train_config
from deepvision.training.engine import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    FitResult,
    TrainingError,
    amp_enabled,
    config_to_dict,
    fit,
    load_checkpoint,
    resolve_device,
    seed_everything,
)

SMOKE_DIR = Path("tmp/smoke")
SMOKE_EXPERIMENT = "deepvision-smoke"
SMOKE_BATCHES = 2
EXIT_OK, EXIT_ERROR, EXIT_INTERRUPTED = 0, 1, 130
HISTORY_FIELDS = [
    "epoch", "phase", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
    "val_macro_precision", "val_macro_recall", "val_macro_f1", "lr_head", "lr_backbone",
    "epoch_seconds", "peak_gpu_memory_mib", "improved",
]  # fmt: skip
BEST_FIELDS = [
    "val_loss",
    "val_accuracy",
    "val_macro_precision",
    "val_macro_recall",
    "val_macro_f1",
]


class CliError(Exception):
    """Raised for invalid command-line usage or unsafe output state."""


# --- Run metadata -----------------------------------------------------------------


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def git_info() -> dict[str, str]:
    """Current commit and whether tracked files have uncommitted changes."""

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    return {"git_commit": commit, "git_dirty": str(dirty).lower()}


def environment_info(device: torch.device, amp: bool) -> dict[str, str]:
    # str(): torch.__version__ is a TorchVersion, which weights_only checkpoint loading rejects.
    info = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "cuda": str(torch.version.cuda),
        "device": str(device),
        "mixed_precision": str(amp).lower(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def config_files(train_config: Path) -> list[Path]:
    """The training config and the data config it references."""
    raw = yaml.safe_load(Path(train_config).read_text(encoding="utf-8"))
    return [Path(train_config), Path(raw["data"]["config"])]


def flatten(values: dict, prefix: str = "") -> dict[str, str]:
    flat = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= flatten(value, f"{name}.")
        else:
            flat[name] = str(value)
    return flat


# --- Outputs ----------------------------------------------------------------------


def prepare_checkpoint_dir(checkpoint_dir: Path, *, resume: bool, restart: bool) -> Path | None:
    """Apply the existing-state policy; return the checkpoint to resume from, if any."""
    best, last = checkpoint_dir / BEST_CHECKPOINT, checkpoint_dir / LAST_CHECKPOINT
    if resume:
        if not last.is_file():
            raise CliError(f"--resume needs {last}, which does not exist")
        return last
    existing = [path for path in (best, last) if path.exists()]
    if existing and not restart:
        names = ", ".join(str(p) for p in existing)
        raise CliError(
            f"training state already exists ({names}); pass --resume to continue it "
            "or --restart to delete it and start over"
        )
    for path in existing:
        path.unlink()
    return None


def write_reports(reports_dir: Path, result: FitResult) -> dict:
    """Write metrics.json (best validation epoch) and history.csv; return the metrics."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    best = next((r for r in result.history if r["epoch"] == result.best_epoch), {})
    peaks = [r["peak_gpu_memory_mib"] for r in result.history if "peak_gpu_memory_mib" in r]
    metrics = {
        "best_epoch": result.best_epoch,
        "best": {field: best.get(field) for field in BEST_FIELDS},
        "epochs_run": len(result.history),
        "stopped_early": result.stopped_early,
        "completed": result.completed,
        "training_seconds": round(sum(r["epoch_seconds"] for r in result.history), 3),
    }
    if peaks:
        metrics["peak_gpu_memory_mib"] = max(peaks)
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (reports_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS, lineterminator="\n", restval="")
        writer.writeheader()
        for record in result.history:
            writer.writerow({k: "" if v is None else v for k, v in record.items()})
    return metrics


# --- MLflow -----------------------------------------------------------------------


def numeric_metrics(record: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in record.items()
        if key != "epoch" and isinstance(value, int | float)  # bools become 0/1
    }


class Tracker:
    """Thin MLflow wrapper; does nothing when disabled."""

    def __init__(self, enabled: bool, tracking_uri: str, experiment: str) -> None:
        self.enabled = enabled
        self.run_id: str | None = None
        if enabled:
            import mlflow

            self.mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)

    def start(self, run_id: str | None, run_name: str, tags: dict, params: dict) -> None:
        """Continue run_id if given (non-empty), otherwise start a new run and log params."""
        if not self.enabled:
            return
        run_id = run_id or None
        run = self.mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name)
        self.run_id = run.info.run_id
        self.mlflow.set_tags(tags)
        if run_id is None:
            self.mlflow.log_params(params)

    def log_epoch(self, record: dict) -> None:
        if self.enabled:
            self.mlflow.log_metrics(numeric_metrics(record), step=record["epoch"])

    def log_artifacts(self, paths: list[Path], checkpoint: Path | None) -> None:
        if not self.enabled:
            return
        for path in paths:
            self.mlflow.log_artifact(str(path))
        if checkpoint is not None:
            self.mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")

    def end(self, status: str) -> None:
        if self.enabled and self.run_id is not None:
            self.mlflow.end_run(status=status)


# --- Main -------------------------------------------------------------------------


def smoke_config(config: TrainConfig) -> TrainConfig:
    """Two epochs, one per phase, writing only under tmp/smoke."""
    fine_tune = replace(config.training.fine_tune, head_epochs=1)
    return replace(
        config,
        training=replace(config.training, epochs=2, fine_tune=fine_tune),
        checkpoint_dir=SMOKE_DIR / "checkpoints",
        reports_dir=SMOKE_DIR / "reports",
        mlflow=replace(config.mlflow, experiment_name=SMOKE_EXPERIMENT),
    )


def print_epoch(record: dict, total_epochs: int) -> None:
    rates = f"lr {record['lr_head']:.2e}"
    if record["lr_backbone"] is not None:
        rates += f"/{record['lr_backbone']:.2e}"
    print(
        f"epoch {record['epoch']}/{total_epochs} [{record['phase']}] "
        f"train loss {record['train_loss']:.4f} acc {record['train_accuracy']:.4f} | "
        f"val loss {record['val_loss']:.4f} acc {record['val_accuracy']:.4f} "
        f"macro-F1 {record['val_macro_f1']:.4f} | {rates} | {record['epoch_seconds']:.0f}s"
        + (" *" if record["improved"] else ""),
        flush=True,
    )


def run(args: argparse.Namespace) -> int:
    config = load_train_config(args.config)
    if args.smoke:
        config = smoke_config(config)
    if args.device:
        config = replace(config, training=replace(config.training, device=args.device))
    max_batches = args.max_batches
    if max_batches is None and args.smoke:
        max_batches = SMOKE_BATCHES

    resume_from = prepare_checkpoint_dir(
        config.checkpoint_dir, resume=args.resume, restart=args.restart or args.smoke
    )
    previous_run_id = None
    if resume_from is not None:
        previous_run_id = load_checkpoint(resume_from)["metadata"].get("mlflow_run_id")

    device = resolve_device(config.training.device)
    amp = amp_enabled(device, config.training.mixed_precision)
    seed_everything(config.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # fixed input size

    try:
        model = build_model(
            config.model.backbone,
            len(config.data.class_names),
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
        )
    except OSError as exc:
        message = f"could not load pretrained weights for {config.model.backbone}: {exc}"
        raise CliError(message) from exc
    loaders = build_dataloaders(config.data, ("train", "val"), device=device)

    git = git_info()
    if git["git_dirty"] == "true":
        print("warning: tracked files have uncommitted changes", file=sys.stderr)
    tags = {
        **git,
        **environment_info(device, amp),
        "splits_md5": file_md5(config.data.manifest),
        "backbone": config.model.backbone,
        "smoke": str(args.smoke).lower(),
        "resumed": str(resume_from is not None).lower(),
    }
    if max_batches is not None:
        tags["max_batches"] = str(max_batches)

    tracker = Tracker(not args.no_mlflow, config.mlflow.tracking_uri, config.mlflow.experiment_name)
    run_name = f"{config.model.backbone}-{git['git_commit'][:7]}"
    tracker.start(previous_run_id, run_name, tags, flatten(config_to_dict(config)))
    try:

        def on_epoch_end(record: dict) -> None:
            print_epoch(record, config.training.epochs)
            tracker.log_epoch(record)

        result = fit(
            model,
            loaders,
            config,
            config.checkpoint_dir,
            device=device,
            resume_from=resume_from,
            metadata={**tags, "mlflow_run_id": tracker.run_id or ""},
            max_epochs=args.max_epochs,
            max_batches=max_batches,
            on_epoch_end=on_epoch_end,
        )
        metrics = write_reports(config.reports_dir, result)
        tracker.log_artifacts(
            [
                config.reports_dir / "metrics.json",
                config.reports_dir / "history.csv",
                *config_files(args.config),
            ],
            result.best_checkpoint if args.log_checkpoint else None,
        )
    except KeyboardInterrupt:
        tracker.end("KILLED")
        raise
    except BaseException:
        tracker.end("FAILED")
        raise
    tracker.end("FINISHED")

    best = metrics["best"]
    print(
        f"best epoch {metrics['best_epoch']}: val macro-F1 {best['val_macro_f1']:.4f}, "
        f"accuracy {best['val_accuracy']:.4f} | epochs run {metrics['epochs_run']}"
        + (" (stopped early)" if metrics["stopped_early"] else "")
    )
    print(f"checkpoints: {config.checkpoint_dir} | reports: {config.reports_dir}")
    if tracker.run_id:
        print(f"MLflow run: {tracker.run_id} ({config.mlflow.experiment_name})")
    return EXIT_OK


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepvision.training.train",
        description="Train the DeepVision classifier on the train and validation splits.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--resume", action="store_true", help="continue from last.pt")
    state.add_argument(
        "--restart", action="store_true", help="delete existing best.pt and last.pt first"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"two tiny epochs ({SMOKE_BATCHES} batches unless --max-batches), under tmp/smoke",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    parser.add_argument(
        "--log-checkpoint", action="store_true", help="also copy best.pt into MLflow"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-epochs", type=positive_int, help="run at most this many epochs now")
    parser.add_argument("--max-batches", type=positive_int, help="limit batches per epoch")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke and args.resume:
        print("error: --smoke cannot be combined with --resume", file=sys.stderr)
        return EXIT_ERROR
    try:
        return run(args)
    except KeyboardInterrupt:
        print("interrupted; continue with --resume", file=sys.stderr)
        return EXIT_INTERRUPTED
    except (CliError, ConfigError, ManifestError, DatasetImageError, TrainingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
