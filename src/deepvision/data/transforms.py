"""Image preprocessing for training and evaluation.

Training images get random crops, horizontal flips, and mild colour jitter.
Validation and test images get deterministic preprocessing only: the shorter side
is resized to the target size and the centre is cropped, which keeps the aspect
ratio of the few non-square images. Both pipelines end with ImageNet normalization
to match pretrained torchvision backbones.
"""

from dataclasses import dataclass

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAIN_SPLITS = frozenset({"train"})
EVAL_SPLITS = frozenset({"val", "test", "test_official"})


@dataclass(frozen=True)
class AugmentationConfig:
    crop_scale: tuple[float, float] = (0.6, 1.0)
    horizontal_flip: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.02


def _finalize() -> list[v2.Transform]:
    return [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def build_transforms(
    split: str, image_size: int = 224, augmentation: AugmentationConfig | None = None
) -> v2.Compose:
    """Return the preprocessing pipeline for a split; only "train" is augmented."""
    if split in TRAIN_SPLITS:
        aug = augmentation or AugmentationConfig()
        steps = [
            v2.ToImage(),
            v2.RandomResizedCrop(image_size, scale=aug.crop_scale, antialias=True),
            v2.RandomHorizontalFlip(aug.horizontal_flip),
            v2.ColorJitter(aug.brightness, aug.contrast, aug.saturation, aug.hue),
        ]
    elif split in EVAL_SPLITS:
        steps = [
            v2.ToImage(),
            v2.Resize(image_size, antialias=True),
            v2.CenterCrop(image_size),
        ]
    else:
        raise ValueError(f"unknown split {split!r}; expected train, val, test, or test_official")
    return v2.Compose(steps + _finalize())
