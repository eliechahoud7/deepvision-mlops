"""PyTorch datasets and data loaders built from the split manifest.

Image paths, labels, and split assignments come from the manifest written by
deepvision.data.split; image paths are relative to the extracted dataset root.
Every split shares one class-to-index mapping taken from the configured class
names, never inferred from the images present in a split.

Loadable splits:

    train          split == "train"            (augmented, shuffled)
    val            split == "val"
    test           split == "test"             clean test set, the primary evaluation
    test_official  split in {test, test_excluded}   all 3,000 original test images

Rows with split "excluded" or "pred" are never loaded.
"""

import csv
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from deepvision.data.transforms import AugmentationConfig, build_transforms

SPLIT_SELECTION = {
    "train": ("train",),
    "val": ("val",),
    "test": ("test",),
    "test_official": ("test", "test_excluded"),
}
MANIFEST_SPLITS = frozenset({"train", "val", "test", "test_excluded", "excluded", "pred"})
LABELED_SPLITS = frozenset({"train", "val", "test", "test_excluded"})
REQUIRED_COLUMNS = ("relpath", "label", "split", "group_id")
# Groups may appear in only one of these; test_excluded counts as test.
_LEAKAGE_SETS = {"train": "train", "val": "val", "test": "test", "test_excluded": "test"}
_MISSING_PREVIEW = 5


class ManifestError(ValueError):
    """Raised when the manifest or configuration is invalid or would leak between splits."""


class DatasetImageError(RuntimeError):
    """Raised when an image listed in the manifest is missing or cannot be decoded."""


def class_to_index(class_names: Sequence[str]) -> dict[str, int]:
    names = list(class_names)
    if not names:
        raise ManifestError("class_names must not be empty")
    if len(set(names)) != len(names):
        raise ManifestError(f"class_names contains duplicates: {names}")
    return {name: index for index, name in enumerate(names)}


def load_manifest(path: Path, class_names: Sequence[str]) -> list[dict]:
    """Read and validate the split manifest; return its rows sorted by path."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"split manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ManifestError(f"manifest {path} is missing columns: {missing}")
        rows = list(reader)

    known = set(class_to_index(class_names))
    seen: set[str] = set()
    group_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        relpath, split, label = row["relpath"], row["split"], row["label"]
        if relpath in seen:
            raise ManifestError(f"duplicate manifest path: {relpath}")
        seen.add(relpath)
        if split not in MANIFEST_SPLITS:
            raise ManifestError(f"unknown split {split!r} for {relpath}")
        if split in LABELED_SPLITS:
            if label not in known:
                raise ManifestError(f"label {label!r} of {relpath} is not in class_names")
            group_sets[row["group_id"]].add(_LEAKAGE_SETS[split])

    leaking = sorted(group for group, sets in group_sets.items() if len(sets) > 1)
    if leaking:
        raise ManifestError(
            f"{len(leaking)} group(s) span train/val/test, e.g. {leaking[:_MISSING_PREVIEW]}"
        )
    return sorted(rows, key=lambda row: row["relpath"])


def load_image(path: Path) -> Image.Image:
    """Decode an image as RGB, raising DatasetImageError with the path on any failure."""
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise DatasetImageError(f"cannot read image {path}: {exc}") from exc


class ManifestImageDataset(Dataset):
    """Images and integer labels for one split of the manifest.

    Holds only paths, labels, and a transform, so it can be sent to spawned
    worker processes on Windows.
    """

    def __init__(
        self,
        rows: list[dict],
        image_root: Path,
        class_names: Sequence[str],
        split: str,
        transform=None,
        check_files: bool = True,
    ) -> None:
        if split not in SPLIT_SELECTION:
            raise ManifestError(f"unknown split {split!r}; expected one of {list(SPLIT_SELECTION)}")
        mapping = class_to_index(class_names)
        selected = SPLIT_SELECTION[split]
        self.split = split
        self.image_root = Path(image_root)
        self.class_names = tuple(class_names)
        self.transform = transform
        self.samples = [
            (row["relpath"], mapping[row["label"]])
            for row in sorted(rows, key=lambda row: row["relpath"])
            if row["split"] in selected
        ]
        if not self.samples:
            raise ManifestError(f"split {split!r} has no images in the manifest")
        if check_files:
            missing = [p for p, _ in self.samples if not (self.image_root / p).is_file()]
            if missing:
                raise DatasetImageError(
                    f"{len(missing)} image(s) for split {split!r} are missing under "
                    f"{self.image_root}, e.g. {missing[:_MISSING_PREVIEW]}"
                )

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        relpath, label = self.samples[index]
        image = load_image(self.image_root / relpath)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass(frozen=True)
class DataConfig:
    manifest: Path
    image_root: Path
    class_names: tuple[str, ...]
    image_size: int = 224
    batch_size: int = 16
    eval_batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 42
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


def load_data_config(train_config: Path = Path("configs/train.yaml")) -> DataConfig:
    """Build the data configuration from train.yaml and the data.yaml it references."""
    train = yaml.safe_load(Path(train_config).read_text(encoding="utf-8"))
    section = train["data"]
    data = yaml.safe_load(Path(section["config"]).read_text(encoding="utf-8"))

    class_names = tuple(section["class_names"])
    class_to_index(class_names)
    if section["num_classes"] != len(class_names):
        raise ManifestError(
            f"num_classes is {section['num_classes']} but {len(class_names)} class_names are listed"
        )
    aug = section.get("augmentation", {})
    jitter = aug.get("color_jitter", {})
    defaults = AugmentationConfig()
    augmentation = AugmentationConfig(
        crop_scale=tuple(aug.get("crop_scale", defaults.crop_scale)),
        horizontal_flip=aug.get("horizontal_flip", defaults.horizontal_flip),
        brightness=jitter.get("brightness", defaults.brightness),
        contrast=jitter.get("contrast", defaults.contrast),
        saturation=jitter.get("saturation", defaults.saturation),
        hue=jitter.get("hue", defaults.hue),
    )
    return DataConfig(
        manifest=Path(data["split"]["output"]),
        image_root=Path(data["archive"]["output"]),
        class_names=class_names,
        image_size=section["image_size"],
        batch_size=train["training"]["batch_size"],
        eval_batch_size=section["eval_batch_size"],
        num_workers=section["num_workers"],
        pin_memory=section["pin_memory"],
        seed=train["seed"],
        augmentation=augmentation,
    )


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_datasets(
    config: DataConfig, splits: Sequence[str] = ("train", "val", "test")
) -> dict[str, ManifestImageDataset]:
    rows = load_manifest(config.manifest, config.class_names)
    return {
        split: ManifestImageDataset(
            rows,
            config.image_root,
            config.class_names,
            split,
            transform=build_transforms(split, config.image_size, config.augmentation),
        )
        for split in splits
    }


def build_dataloaders(
    config: DataConfig,
    splits: Sequence[str] = ("train", "val", "test"),
    device: str | torch.device = "auto",
) -> dict[str, DataLoader]:
    """Create loaders: the train split is shuffled with a seeded generator, others keep order."""
    pin_memory = config.pin_memory and _resolve_device(device).type == "cuda"
    workers = config.num_workers
    loaders = {}
    for split, dataset in build_datasets(config, splits).items():
        common = {
            "num_workers": workers,
            "pin_memory": pin_memory,
            "persistent_workers": workers > 0,  # avoids re-spawning workers each epoch
        }
        if split == "train":
            loaders[split] = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(config.seed),
                # Drop the last partial batch only when at least one full batch exists.
                drop_last=len(dataset) >= config.batch_size,
                **common,
            )
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=config.eval_batch_size, shuffle=False, **common
            )
    return loaders
