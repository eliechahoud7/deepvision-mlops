import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from deepvision.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from deepvision.models.factory import build_model
from deepvision.training.engine import CHECKPOINT_VERSION, save_checkpoint

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


def make_checkpoint(path: Path, *, image_size: int = 32, seed: int = 0, **overrides) -> Path:
    """Write a checkpoint in the trainer's format with random (not pretrained) weights."""
    torch.manual_seed(seed)
    model = build_model("mobilenet_v3_small", len(CLASSES), pretrained=False)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "epoch": 3,
        "phase": "full",
        "model_state": model.state_dict(),
        "backbone": "mobilenet_v3_small",
        "num_classes": len(CLASSES),
        "class_names": list(CLASSES),
        "class_to_index": {name: i for i, name in enumerate(CLASSES)},
        "image_size": image_size,
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "metrics": {
            "loss": 0.5, "accuracy": 0.8, "macro_precision": 0.81, "macro_recall": 0.79,
            "macro_f1": 0.8, "confusion_matrix": [[0] * 6 for _ in range(6)],
        },
        "config": {},
        "metadata": {"git_commit": "a" * 40, "git_dirty": "false", "mlflow_run_id": "run-1"},
    }  # fmt: skip
    payload |= overrides
    save_checkpoint(path, payload)
    return path


def image_bytes(
    fmt: str = "PNG", size: tuple[int, int] = (40, 30), mode: str = "RGB", seed: int = 0, **save
) -> bytes:
    """Encode a random image of the given mode and format."""
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels).convert(mode)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save)
    return buffer.getvalue()


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    return make_checkpoint(tmp_path / "best.pt")
