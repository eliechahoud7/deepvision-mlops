"""Training engine: two-phase fine-tuning, validation metrics, early stopping, checkpoints.

Phase "head" trains only the classification head while the backbone is frozen (and kept
in eval mode by ImageClassifier). Phase "full" unfreezes the backbone and trains the
whole network with separate learning rates for backbone and head. Each phase gets a
fresh AdamW optimizer and its own cosine schedule, stepped once per optimizer step.

fit() only accepts train and validation loaders: early stopping and best-model
selection use validation metrics exclusively. Test evaluation is a separate stage.

Checkpoints contain only tensors and plain Python values, so they load with
torch.load(..., weights_only=True):

    best.pt   model weights and metadata from the best validation epoch
    last.pt   everything needed to resume: also optimizer, scheduler, gradient scaler,
              early-stopping state, history, and random number generator states
"""

import math
import os
import random
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from deepvision.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from deepvision.models.factory import ImageClassifier
from deepvision.training.config import TrainConfig, TrainingConfig

CHECKPOINT_VERSION = 1
BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"
ALLOWED_LOADERS = frozenset({"train", "val"})
# Early-stopping monitor -> key in the validation metrics.
MONITOR_KEYS = {"val_macro_f1": "macro_f1", "val_accuracy": "accuracy", "val_loss": "loss"}


class TrainingError(RuntimeError):
    """Raised when training cannot continue."""


class NonFiniteLossError(TrainingError):
    """Raised when a training or validation loss is NaN or infinite."""


# --- Environment ------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python and PyTorch (CPU and all CUDA devices)."""
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("training.device is cuda but CUDA is not available")
    return resolved


def amp_enabled(device: torch.device, mixed_precision: bool) -> bool:
    """FP16 autocast is used only on CUDA; CPU always trains in full precision."""
    return mixed_precision and device.type == "cuda"


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device.type, dtype=torch.float16) if amp else nullcontext()


# --- Metrics ----------------------------------------------------------------------


def confusion_matrix(
    targets: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Counts with rows = true class and columns = predicted class."""
    index = targets.long() * num_classes + predictions.long()
    return torch.bincount(index, minlength=num_classes**2).reshape(num_classes, num_classes)


def classification_metrics(matrix: torch.Tensor) -> dict[str, float]:
    """Accuracy and macro-averaged precision, recall, and F1; undefined ratios count as 0."""
    counts = matrix.double()
    true_positive = counts.diagonal()
    predicted, actual = counts.sum(dim=0), counts.sum(dim=1)
    precision = torch.where(predicted > 0, true_positive / predicted.clamp(min=1), 0.0)
    recall = torch.where(actual > 0, true_positive / actual.clamp(min=1), 0.0)
    total = precision + recall
    f1 = torch.where(total > 0, 2 * precision * recall / total.clamp(min=1e-12), 0.0)
    return {
        "accuracy": (true_positive.sum() / counts.sum().clamp(min=1)).item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }


# --- Training and evaluation ------------------------------------------------------


def _optimizer_step(optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler) -> bool:
    """Step through the gradient scaler; return False if it skipped the step (inf grads)."""
    scale_before = scaler.get_scale() if scaler.is_enabled() else None
    scaler.step(optimizer)
    scaler.update()
    return scale_before is None or scaler.get_scale() >= scale_before


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp: bool = False,
    accumulation_steps: int = 1,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_batches: int | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    """Train for one epoch with gradient accumulation.

    Each loss is divided by the number of batches in its accumulation window (the last
    window may be shorter), so a window's gradient equals the mean over its batches.
    The optimizer and scheduler step once per window.
    """
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    if total == 0:
        raise TrainingError("the training loader produced no batches")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, correct, seen, steps = 0.0, 0, 0, 0
    for index, (images, targets) in enumerate(loader):
        if index >= total:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise NonFiniteLossError(
                f"non-finite training loss ({loss.item()}) in epoch {epoch}, batch {index + 1}"
            )
        window_start = index - index % accumulation_steps
        window = min(accumulation_steps, total - window_start)
        scaler.scale(loss / window).backward()

        batch_size = targets.numel()
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size
        if index + 1 - window_start == window:
            if _optimizer_step(optimizer, scaler) and scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    return {"loss": loss_sum / seen, "accuracy": correct / seen, "optimizer_steps": steps}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    amp: bool = False,
    max_batches: int | None = None,
) -> dict:
    """Loss, accuracy, macro precision/recall/F1, and the confusion matrix (as a list)."""
    model.eval()
    loss_sum, seen = 0.0, 0
    all_targets, all_predictions = [], []
    for index, (images, targets) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device, amp):
            logits = model(images)
        loss_sum += F.cross_entropy(logits.float(), targets, reduction="sum").item()
        seen += targets.numel()
        all_targets.append(targets.cpu())
        all_predictions.append(logits.argmax(dim=1).cpu())
    if seen == 0:
        raise TrainingError("the validation loader produced no batches")
    loss = loss_sum / seen
    if not math.isfinite(loss):
        raise NonFiniteLossError(f"non-finite validation loss ({loss})")
    matrix = confusion_matrix(torch.cat(all_targets), torch.cat(all_predictions), num_classes)
    return {"loss": loss, **classification_metrics(matrix), "confusion_matrix": matrix.tolist()}


# --- Early stopping ---------------------------------------------------------------


@dataclass
class EarlyStopping:
    """Stop after `patience` epochs without strict improvement of the monitored metric."""

    monitor: str
    mode: str
    patience: int
    best: float | None = None
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, value: float, epoch: int) -> bool:
        """Record an epoch's value; return True if it is a new best."""
        improved = math.isfinite(value) and (
            self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        )
        if improved:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state: Mapping) -> None:
        for key, value in state.items():
            setattr(self, key, value)


# --- Optimizer and scheduler ------------------------------------------------------


def phase_for_epoch(epoch_index: int, head_epochs: int) -> str:
    """Phase for a 0-based epoch index: "head" for the first head_epochs, then "full"."""
    return "head" if epoch_index < head_epochs else "full"


def build_optimizer(
    model: ImageClassifier, phase: str, training: TrainingConfig
) -> torch.optim.Optimizer:
    fine_tune = training.fine_tune
    groups = [
        {"params": model.head_parameters(), "lr": fine_tune.head_learning_rate, "name": "head"}
    ]
    if phase == "full":
        backbone = {
            "params": model.backbone_parameters(),
            "lr": fine_tune.backbone_learning_rate,
            "name": "backbone",
        }
        groups.insert(0, backbone)
    return torch.optim.AdamW(groups, weight_decay=training.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, training: TrainingConfig, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Cosine decay to zero over a phase's optimizer steps, or None for a constant rate."""
    if training.scheduler == "none":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))


# --- Checkpoints ------------------------------------------------------------------


def _plain(value):
    """Convert config values to types torch.load(weights_only=True) accepts."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def config_to_dict(config: TrainConfig) -> dict:
    return _plain(asdict(config))


def save_checkpoint(path: Path, payload: dict) -> None:
    """Write atomically, so an interruption never leaves a truncated checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location, weights_only=True)


def _model_payload(
    model: ImageClassifier,
    config: TrainConfig,
    epoch: int,
    phase: str,
    metrics: dict,
    metadata: Mapping | None,
) -> dict:
    class_names = list(config.data.class_names)
    return {
        "format_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "phase": phase,
        "model_state": model.state_dict(),
        "backbone": model.backbone,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_to_index": {name: index for index, name in enumerate(class_names)},
        "image_size": config.data.image_size,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "metrics": metrics,
        "config": config_to_dict(config),
        "metadata": _plain(dict(metadata or {})),
    }


def _rng_state(device: torch.device, train_loader: DataLoader) -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    generator = getattr(train_loader, "generator", None)
    if generator is not None:
        state["loader_generator"] = generator.get_state()
    return state


def _restore_rng(state: Mapping, device: torch.device, train_loader: DataLoader) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator = getattr(train_loader, "generator", None)
    if generator is not None and "loader_generator" in state:
        generator.set_state(state["loader_generator"])


def _check_resume(state: Mapping, model: ImageClassifier, config: TrainConfig) -> None:
    current = config_to_dict(config)
    problems = []
    if state.get("format_version") != CHECKPOINT_VERSION:
        problems.append(f"checkpoint format {state.get('format_version')}")
    if state["backbone"] != model.backbone:
        problems.append(f"backbone {state['backbone']} != {model.backbone}")
    if state["class_names"] != list(config.data.class_names):
        problems.append("class names differ")
    for section in ("model", "training"):
        if state["config"][section] != current[section]:
            problems.append(f"{section} configuration differs")
    if problems:
        raise TrainingError(f"cannot resume from this checkpoint: {'; '.join(problems)}")


# --- Fit --------------------------------------------------------------------------


@dataclass
class FitResult:
    history: list[dict]
    best_epoch: int
    best_metric: float | None
    stopped_early: bool
    completed: bool
    best_checkpoint: Path | None
    last_checkpoint: Path | None


def _check_loaders(loaders: Mapping[str, DataLoader]) -> None:
    if set(loaders) != ALLOWED_LOADERS:
        raise ValueError(
            f"fit() takes exactly 'train' and 'val' loaders, got {sorted(loaders)}; "
            "the test set must not be used during training or model selection"
        )
    for name, loader in loaders.items():
        split = getattr(loader.dataset, "split", None)
        if split is not None and split != name:
            raise ValueError(f"the {name!r} loader contains the {split!r} split")


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float | None]:
    rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
    return {"lr_head": rates["head"], "lr_backbone": rates.get("backbone")}


def fit(
    model: ImageClassifier,
    loaders: Mapping[str, DataLoader],
    config: TrainConfig,
    checkpoint_dir: Path,
    *,
    device: str | torch.device | None = None,
    resume_from: Path | None = None,
    metadata: Mapping | None = None,
    max_epochs: int | None = None,
    max_batches: int | None = None,
    on_epoch_end: Callable[[dict], None] | None = None,
) -> FitResult:
    """Run two-phase training with validation, early stopping, and checkpointing.

    max_epochs limits how many epochs this call runs (training can be resumed later);
    max_batches limits batches per epoch for quick checks. on_epoch_end receives each
    epoch's record, for example to log it to an experiment tracker.
    """
    _check_loaders(loaders)
    training = config.training
    head_epochs = training.fine_tune.head_epochs
    stopping = training.early_stopping
    num_classes = len(config.data.class_names)
    device = resolve_device(device if device is not None else training.device)
    amp = amp_enabled(device, training.mixed_precision)
    checkpoint_dir = Path(checkpoint_dir)
    best_path, last_path = checkpoint_dir / BEST_CHECKPOINT, checkpoint_dir / LAST_CHECKPOINT

    model.to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    stopper = EarlyStopping(stopping.monitor, stopping.mode, stopping.patience)
    history: list[dict] = []
    start_epoch, resumed = 0, None
    if resume_from is not None:
        resumed = load_checkpoint(resume_from)
        _check_resume(resumed, model, config)
        model.load_state_dict(resumed["model_state"])
        history = list(resumed["history"])
        start_epoch = resumed["epoch"]
        stopper.load_state_dict(resumed["early_stopping"])
        scaler.load_state_dict(resumed["scaler_state"])
        _restore_rng(resumed["rng"], device, loaders["train"])
    # A fresh run never reports a best.pt left over from an earlier run.
    best_saved = resumed is not None and best_path.exists()
    if resumed is not None and resumed["completed"]:
        return FitResult(
            history,
            stopper.best_epoch,
            stopper.best,
            resumed["stopped_early"],
            True,
            best_path if best_saved else None,
            Path(resume_from),
        )

    batches = len(loaders["train"])
    if max_batches is not None:
        batches = min(batches, max_batches)
    steps_per_epoch = math.ceil(batches / training.gradient_accumulation_steps)
    end_epoch = training.epochs
    if max_epochs is not None:
        end_epoch = min(end_epoch, start_epoch + max_epochs)
    optimizer = scheduler = phase = None
    stopped_early = completed = False
    last_saved = None

    for epoch in range(start_epoch, end_epoch):
        epoch_phase = phase_for_epoch(epoch, head_epochs)
        if epoch_phase != phase:
            phase = epoch_phase
            model.set_backbone_trainable(phase == "full")
            optimizer = build_optimizer(model, phase, training)
            phase_epochs = head_epochs if phase == "head" else training.epochs - head_epochs
            scheduler = build_scheduler(optimizer, training, steps_per_epoch * phase_epochs)
            # Continue the same phase's optimizer and schedule; a new phase starts fresh.
            if resumed is not None and epoch == start_epoch and resumed["phase"] == phase:
                optimizer.load_state_dict(resumed["optimizer_state"])
                if scheduler is not None:
                    scheduler.load_state_dict(resumed["scheduler_state"])

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            train_metrics = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                device=device,
                scaler=scaler,
                amp=amp,
                accumulation_steps=training.gradient_accumulation_steps,
                scheduler=scheduler,
                max_batches=max_batches,
                epoch=epoch + 1,
            )
            val_metrics = evaluate(
                model,
                loaders["val"],
                device=device,
                num_classes=num_classes,
                amp=amp,
                max_batches=max_batches,
            )
        except torch.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise TrainingError(
                f"CUDA ran out of memory in epoch {epoch + 1} with batch_size "
                f"{training.batch_size}; lower training.batch_size and raise "
                "training.gradient_accumulation_steps to keep the effective batch size"
            ) from exc

        record = {
            "epoch": epoch + 1,
            "phase": phase,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
            **_learning_rates(optimizer),
            "epoch_seconds": round(time.perf_counter() - started, 3),
        }
        if device.type == "cuda":
            record["peak_gpu_memory_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20)
        record["improved"] = stopper.update(val_metrics[MONITOR_KEYS[stopping.monitor]], epoch + 1)
        history.append(record)

        payload = _model_payload(model, config, epoch + 1, phase, val_metrics, metadata)
        if record["improved"]:
            save_checkpoint(best_path, payload)
            best_saved = True
        stopped_early = stopper.should_stop
        completed = stopped_early or epoch + 1 == training.epochs
        save_checkpoint(
            last_path,
            payload
            | {
                "history": history,
                "completed": completed,
                "stopped_early": stopped_early,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict(),
                "early_stopping": stopper.state_dict(),
                "rng": _rng_state(device, loaders["train"]),
            },
        )
        last_saved = last_path
        if on_epoch_end is not None:
            on_epoch_end(record)
        if stopped_early:
            break

    return FitResult(
        history,
        stopper.best_epoch,
        stopper.best,
        stopped_early,
        completed,
        best_path if best_saved else None,
        last_saved or (Path(resume_from) if resume_from is not None else None),
    )
