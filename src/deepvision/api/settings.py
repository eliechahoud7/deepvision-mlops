"""Service settings read from DEEPVISION_* environment variables."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from deepvision.inference.predictor import DEFAULT_MAX_PIXELS

PREFIX = "DEEPVISION_"
DEVICES = frozenset({"auto", "cpu", "cuda"})


class SettingsError(ValueError):
    """Raised when an environment variable has an invalid value."""


@dataclass(frozen=True)
class Settings:
    checkpoint: Path = Path("artifacts/checkpoints/best.pt")
    evaluation_metrics: Path = Path("reports/evaluation/metrics.json")
    device: str = "cpu"
    max_upload_mb: float = 10.0
    max_batch: int = 16
    top_k: int = 3
    max_pixels: int = DEFAULT_MAX_PIXELS
    num_threads: int | None = None  # None keeps PyTorch's default

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise SettingsError(f"{PREFIX}DEVICE must be one of {sorted(DEVICES)}")
        for name in ("max_upload_mb", "max_batch", "top_k", "max_pixels"):
            if getattr(self, name) <= 0:
                raise SettingsError(f"{PREFIX}{name.upper()} must be positive")
        if self.num_threads is not None and self.num_threads < 1:
            raise SettingsError(f"{PREFIX}NUM_THREADS must be at least 1")

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        environ = os.environ if environ is None else environ
        values = {}
        for field in fields(cls):
            raw = environ.get(PREFIX + field.name.upper())
            if raw is None or raw == "":
                continue
            try:
                if field.name in ("checkpoint", "evaluation_metrics"):
                    values[field.name] = Path(raw)
                elif field.name == "device":
                    values[field.name] = raw.strip().lower()
                elif field.name == "max_upload_mb":
                    values[field.name] = float(raw)
                else:
                    values[field.name] = int(raw)
            except ValueError as exc:
                raise SettingsError(f"{PREFIX}{field.name.upper()} is not a valid number") from exc
        return cls(**values)
