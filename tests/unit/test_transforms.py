import numpy as np
import pytest
import torch
from PIL import Image

from deepvision.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    AugmentationConfig,
    build_transforms,
)


def image(width: int = 40, height: int = 30, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.parametrize("split", ["val", "test", "test_official"])
def test_eval_preprocessing_is_deterministic(split):
    transform = build_transforms(split, image_size=32)
    first, second = transform(image()), transform(image())

    assert first.shape == (3, 32, 32)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_eval_keeps_aspect_ratio_then_center_crops():
    # A uniform image stays uniform, so the normalized value is known exactly.
    gray = Image.new("RGB", (60, 30), (128, 128, 128))
    out = build_transforms("val", image_size=24)(gray)

    assert out.shape == (3, 24, 24)
    expected = (128 / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    assert torch.allclose(out.mean(dim=(1, 2)), expected, atol=1e-4)


def test_train_augmentation_varies_and_is_reproducible_with_a_seed():
    transform = build_transforms("train", image_size=32)
    source = image(seed=1)

    torch.manual_seed(0)
    a, b = transform(source), transform(source)
    torch.manual_seed(0)
    a_again = transform(source)

    assert a.shape == (3, 32, 32) and a.dtype == torch.float32
    assert not torch.equal(a, b)
    assert torch.equal(a, a_again)


def test_augmentation_is_configurable():
    no_aug = AugmentationConfig(
        crop_scale=(1.0, 1.0), horizontal_flip=0.0, brightness=0, contrast=0, saturation=0, hue=0
    )
    square = image(32, 32, seed=2)
    train = build_transforms("train", image_size=32, augmentation=no_aug)

    assert torch.allclose(train(square), build_transforms("val", image_size=32)(square), atol=1e-6)


def test_only_train_is_augmented():
    train = build_transforms("train")
    val = build_transforms("val")
    train_types = {type(t).__name__ for t in train.transforms}
    val_types = {type(t).__name__ for t in val.transforms}

    assert {"RandomResizedCrop", "RandomHorizontalFlip", "ColorJitter"} <= train_types
    assert not val_types & {"RandomResizedCrop", "RandomHorizontalFlip", "ColorJitter"}
    assert {"Resize", "CenterCrop", "Normalize"} <= val_types


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        build_transforms("pred")
