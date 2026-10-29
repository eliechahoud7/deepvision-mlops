"""Load a trained checkpoint into an evaluation-ready model.

Shared by held-out evaluation and the inference service so both measure and serve
exactly the same model: the network is rebuilt from the checkpoint's own metadata
(backbone, number of classes) with pretrained=False, so nothing is downloaded, and the
checkpoint is read with torch.load(weights_only=True).

CheckpointError.code is a short, path-free reason suitable for API responses; the
message itself may contain the checkpoint path and is meant for local CLI output.
"""

import pickle
from pathlib import Path

import torch

from deepvision.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from deepvision.models.factory import ImageClassifier, build_model
from deepvision.training.engine import CHECKPOINT_VERSION, load_checkpoint

REQUIRED_KEYS = (
    "format_version", "model_state", "backbone", "num_classes", "class_names",
    "image_size", "normalization",
)  # fmt: skip


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, unreadable, or does not match the project."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def load_trained_model(
    path: Path,
    *,
    class_names: tuple[str, ...] | list[str] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[ImageClassifier, dict]:
    """Rebuild the trained model in eval mode and return it with the checkpoint dict.

    If class_names is given, the checkpoint must list exactly those classes in order.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError("not_found", f"checkpoint not found: {path}; run `dvc repro train`")
    try:
        checkpoint = load_checkpoint(path)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        raise CheckpointError("unreadable", f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or any(key not in checkpoint for key in REQUIRED_KEYS):
        raise CheckpointError("invalid", f"checkpoint {path} is missing required fields")
    if checkpoint["format_version"] != CHECKPOINT_VERSION:
        raise CheckpointError(
            "unsupported_format", f"unsupported checkpoint format {checkpoint['format_version']}"
        )
    if class_names is not None and checkpoint["class_names"] != list(class_names):
        raise CheckpointError(
            "class_mismatch",
            f"checkpoint classes {checkpoint['class_names']} differ from the configured "
            f"class_names {list(class_names)}",
        )
    expected = {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)}
    if checkpoint["normalization"] != expected:
        raise CheckpointError(
            "unsupported_normalization",
            f"checkpoint normalization {checkpoint['normalization']} is not ImageNet's",
        )
    try:
        model = build_model(checkpoint["backbone"], checkpoint["num_classes"], pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
    except (ValueError, RuntimeError, KeyError) as exc:
        raise CheckpointError("invalid", f"checkpoint {path} weights do not fit its model") from exc
    return model.to(device).eval(), checkpoint
