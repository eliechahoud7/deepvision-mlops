"""Image decoding and model inference for serving, independent of any web framework.

The predictor loads the trained checkpoint through the same helper as held-out
evaluation and preprocesses images with the same deterministic evaluation transform,
so served predictions come from exactly the evaluated model. Inference runs in eval
mode, under torch.inference_mode(), in FP32.

decode_image() trusts the image content, never the filename or declared type: only
JPEG, PNG, WebP, and BMP are accepted, the pixel count is checked from the header
before decoding, EXIF orientation is applied, and every image is converted to RGB.
"""

import hashlib
import io
import json
import threading
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from deepvision.data.transforms import build_transforms
from deepvision.evaluation.checkpoint import load_trained_model
from deepvision.training.engine import resolve_device

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP"})
DEFAULT_MAX_PIXELS = 40_000_000  # ~ a 7,000 x 5,700 photo; training images are 150 x 150


class ImageError(ValueError):
    """Base class for rejected images; `code` is a stable, client-safe identifier."""

    code = "invalid_image"


class InvalidImageError(ImageError):
    code = "invalid_image"


class UnsupportedFormatError(ImageError):
    code = "unsupported_format"


class ImageTooLargeError(ImageError):
    code = "image_too_large"


def decode_image(data: bytes, *, max_pixels: int = DEFAULT_MAX_PIXELS) -> Image.Image:
    """Decode uploaded bytes into an upright RGB image, or raise an ImageError."""
    if not data:
        raise InvalidImageError("the uploaded file is empty")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            if image.format not in ALLOWED_FORMATS:
                raise UnsupportedFormatError(
                    f"unsupported image format {image.format}; "
                    f"expected one of {sorted(ALLOWED_FORMATS)}"
                )
            width, height = image.size
            if width * height > max_pixels:
                raise ImageTooLargeError(
                    f"image has {width * height} pixels; the limit is {max_pixels}"
                )
            image.load()
    except ImageError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError("image dimensions exceed the decompression limit") from exc
    except UnidentifiedImageError as exc:
        raise InvalidImageError("the uploaded file is not a recognizable image") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("the image data is corrupt or truncated") from exc
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # malformed EXIF: keep the stored orientation rather than fail
        pass
    return image.convert("RGB")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]
    probabilities: dict[str, float]


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_metrics(path: Path | None, checkpoint_md5: str) -> dict | None:
    """Held-out metrics from reports/evaluation, only if they belong to this checkpoint."""
    if path is None or not Path(path).is_file():
        return None
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if report.get("checkpoint", {}).get("md5") != checkpoint_md5:
        return None  # stale: produced by a different checkpoint
    keys = ("images", "accuracy", "macro_f1", "macro_precision", "macro_recall")
    return {
        split: {key: report[split][key] for key in keys if key in report[split]}
        for split in ("test", "test_official")
        if split in report
    }


class Predictor:
    """Loads the checkpoint once and serves thread-safe batched predictions."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "cpu",
        top_k: int = 3,
        max_batch: int = 16,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        num_threads: int | None = None,
        evaluation_metrics: Path | None = None,
    ) -> None:
        if top_k < 1 or max_batch < 1 or max_pixels < 1:
            raise ValueError("top_k, max_batch, and max_pixels must be at least 1")
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self.device = resolve_device(device)
        self.model, checkpoint = load_trained_model(checkpoint_path, device=self.device)
        self.class_names: list[str] = list(checkpoint["class_names"])
        self.image_size: int = checkpoint["image_size"]
        self.transform = build_transforms("test", self.image_size)
        self.top_k = min(top_k, len(self.class_names))
        self.max_batch = max_batch
        self.max_pixels = max_pixels
        # One forward pass at a time: avoids threads oversubscribing CPU cores.
        self._lock = threading.Lock()

        checkpoint_md5 = file_md5(checkpoint_path)
        metadata = checkpoint.get("metadata", {})
        self.version = f"{checkpoint['backbone']}-{checkpoint_md5[:12]}"
        self.info = {
            "model_version": self.version,
            "backbone": checkpoint["backbone"],
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
            "image_size": self.image_size,
            "epoch": checkpoint.get("epoch"),
            "checkpoint_md5": checkpoint_md5,
            "training_commit": metadata.get("git_commit", "unknown"),
            "training_tree_dirty": metadata.get("git_dirty", "unknown"),
            "mlflow_run_id": metadata.get("mlflow_run_id") or None,
            "device": str(self.device),
            "validation_metrics": {
                key: value
                for key, value in checkpoint.get("metrics", {}).items()
                if key != "confusion_matrix"
            },
            "evaluation_metrics": _evaluation_metrics(evaluation_metrics, checkpoint_md5),
        }

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)

    def decode(self, data: bytes) -> Image.Image:
        return decode_image(data, max_pixels=self.max_pixels)

    def predict(self, images: Sequence[Image.Image]) -> tuple[list[Prediction], float]:
        """Predict in request order; returns predictions and inference time in ms."""
        if not images:
            return [], 0.0
        started = time.perf_counter()
        batch = torch.stack([self.preprocess(image) for image in images])
        chunks = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(batch), self.max_batch):
                chunk = batch[start : start + self.max_batch].to(self.device)
                chunks.append(torch.softmax(self.model(chunk).float(), dim=1).cpu())
        probabilities = torch.cat(chunks)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return [self._to_prediction(row) for row in probabilities], elapsed_ms

    def _to_prediction(self, probabilities: torch.Tensor) -> Prediction:
        values, indices = probabilities.topk(self.top_k)
        top = [(self.class_names[i], float(v)) for v, i in zip(values, indices, strict=True)]
        return Prediction(
            label=top[0][0],
            confidence=top[0][1],
            top_k=top,
            probabilities={
                name: float(p) for name, p in zip(self.class_names, probabilities, strict=True)
            },
        )
