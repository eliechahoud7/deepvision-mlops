import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from deepvision.data.dataset import (
    DataConfig,
    DatasetImageError,
    ManifestError,
    ManifestImageDataset,
    build_dataloaders,
    class_to_index,
    load_data_config,
    load_manifest,
)
from deepvision.data.transforms import build_transforms

CLASSES = ("buildings", "forest", "sea")
FIELDS = ["relpath", "source_split", "label", "split", "group_id", "group_size", "reasons"]
REPO_ROOT = Path(__file__).resolve().parents[2]


def row(relpath: str, split: str, label: str = "", group: str | None = None) -> dict:
    source = relpath.split("/")[0]
    return {
        "relpath": relpath,
        "source_split": source,
        "label": label,
        "split": split,
        "group_id": group or relpath,
        "group_size": "1",
        "reasons": "",
    }


def default_rows() -> list[dict]:
    rows = []
    for label in CLASSES:
        rows += [row(f"train/{label}/{i}.jpg", "train", label) for i in range(4)]
        rows.append(row(f"train/{label}/v.jpg", "val", label))
        rows.append(row(f"test/{label}/t.jpg", "test", label))
    rows += [
        row("test/sea/x.jpg", "test_excluded", "sea"),
        row("train/forest/excluded.jpg", "excluded", "forest"),
        row("pred/1.jpg", "pred"),
    ]
    return rows


def write_manifest(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_images(root: Path, rows: list[dict]) -> None:
    rng = np.random.default_rng(0)
    for r in rows:
        path = root / r["relpath"]
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(path, format="JPEG")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "images"
    rows = default_rows()
    write_images(root, rows)
    return write_manifest(tmp_path / "splits.csv", rows), root


def config(manifest: Path, root: Path, **overrides) -> DataConfig:
    values = {
        "manifest": manifest,
        "image_root": root,
        "class_names": CLASSES,
        "image_size": 32,
        "batch_size": 4,
        "eval_batch_size": 2,
        "num_workers": 0,
        "seed": 42,
    }
    return DataConfig(**(values | overrides))


# --- Manifest validation -------------------------------------------------------------


def test_manifest_is_loaded_sorted(dataset_dir):
    manifest, _ = dataset_dir
    rows = load_manifest(manifest, CLASSES)
    assert [r["relpath"] for r in rows] == sorted(r["relpath"] for r in default_rows())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate manifest path"),
        (lambda rows: rows[0].update(split="holdout"), "unknown split"),
        (lambda rows: rows[0].update(label="desert"), "not in class_names"),
        (lambda rows: rows[0].update(label=""), "not in class_names"),
    ],
)
def test_invalid_manifest_rows_are_rejected(tmp_path, mutate, message):
    rows = default_rows()
    mutate(rows)
    with pytest.raises(ManifestError, match=message):
        load_manifest(write_manifest(tmp_path / "m.csv", rows), CLASSES)


def test_missing_columns_and_files_are_rejected(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "missing.csv", CLASSES)
    path = write_manifest(tmp_path / "m.csv", default_rows(), ["relpath", "label", "split"])
    with pytest.raises(ManifestError, match="missing columns"):
        load_manifest(path, CLASSES)


@pytest.mark.parametrize(
    ("first", "second"), [("train", "test"), ("train", "val"), ("val", "test_excluded")]
)
def test_groups_spanning_splits_are_leakage(tmp_path, first, second):
    rows = default_rows() + [
        row("train/sea/a.jpg", first, "sea", group="g1"),
        row("train/sea/b.jpg", second, "sea", group="g1"),
    ]
    with pytest.raises(ManifestError, match="span train/val/test"):
        load_manifest(write_manifest(tmp_path / "m.csv", rows), CLASSES)


def test_excluded_and_pred_groups_are_not_leakage(tmp_path):
    rows = default_rows() + [
        row("train/sea/a.jpg", "train", "sea", group="g1"),
        row("train/sea/b.jpg", "excluded", "sea", group="g1"),
        row("pred/9.jpg", "pred", group="g1"),
    ]
    assert load_manifest(write_manifest(tmp_path / "m.csv", rows), CLASSES)


@pytest.mark.parametrize("names", [(), ("sea", "sea")])
def test_invalid_class_names_are_rejected(names):
    with pytest.raises(ManifestError, match="class_names"):
        class_to_index(names)


# --- Dataset ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("split", "expected"), [("train", 12), ("val", 3), ("test", 3), ("test_official", 4)]
)
def test_split_selection(dataset_dir, split, expected):
    manifest, root = dataset_dir
    dataset = ManifestImageDataset(load_manifest(manifest, CLASSES), root, CLASSES, split)

    assert len(dataset) == expected
    paths = [p for p, _ in dataset.samples]
    assert not any("excluded.jpg" in p or p.startswith("pred/") for p in paths)
    assert paths == sorted(paths)


def test_class_mapping_is_shared_across_splits(tmp_path):
    rows = [
        row("train/buildings/1.jpg", "train", "buildings"),
        row("train/sea/1.jpg", "train", "sea"),
        row("val/sea/1.jpg", "val", "sea"),  # val has no buildings or forest images
    ]
    write_images(tmp_path, rows)
    loaded = load_manifest(write_manifest(tmp_path / "m.csv", rows), CLASSES)

    train = ManifestImageDataset(loaded, tmp_path, CLASSES, "train")
    val = ManifestImageDataset(loaded, tmp_path, CLASSES, "val")

    assert train.targets == [0, 2]
    assert val.targets == [2]
    assert train.class_names == val.class_names == CLASSES


def test_empty_or_unknown_split_is_rejected(tmp_path):
    rows = [row("train/sea/1.jpg", "train", "sea")]
    with pytest.raises(ManifestError, match="no images"):
        ManifestImageDataset(rows, tmp_path, CLASSES, "val", check_files=False)
    with pytest.raises(ManifestError, match="unknown split"):
        ManifestImageDataset(rows, tmp_path, CLASSES, "pred", check_files=False)


def test_missing_images_are_reported_when_the_dataset_is_built(dataset_dir):
    manifest, root = dataset_dir
    (root / "test" / "sea" / "t.jpg").unlink()

    with pytest.raises(DatasetImageError, match=r"1 image\(s\) for split 'test'.*test/sea/t.jpg"):
        ManifestImageDataset(load_manifest(manifest, CLASSES), root, CLASSES, "test")


def test_unreadable_image_is_reported_with_its_path(dataset_dir):
    manifest, root = dataset_dir
    (root / "train" / "sea" / "v.jpg").write_bytes(b"not an image")
    dataset = ManifestImageDataset(load_manifest(manifest, CLASSES), root, CLASSES, "val")
    index = [p for p, _ in dataset.samples].index("train/sea/v.jpg")

    with pytest.raises(DatasetImageError, match=r"cannot read image .*v\.jpg"):
        dataset[index]


def test_items_are_normalized_tensors_with_int_labels(dataset_dir):
    manifest, root = dataset_dir
    gray = root / "test" / "forest" / "t.jpg"
    Image.new("L", (10, 20), 90).save(gray, format="JPEG")  # greyscale becomes 3 channels
    dataset = ManifestImageDataset(
        load_manifest(manifest, CLASSES), root, CLASSES, "test", build_transforms("test", 32)
    )

    image, label = dataset[[p for p, _ in dataset.samples].index("test/forest/t.jpg")]
    assert image.shape == (3, 32, 32) and image.dtype == torch.float32
    assert label == CLASSES.index("forest")


# --- Data loaders -------------------------------------------------------------------


def test_loaders_produce_batches(dataset_dir):
    loaders = build_dataloaders(config(*dataset_dir), ("train", "val", "test"), device="cpu")

    images, labels = next(iter(loaders["train"]))
    assert images.shape == (4, 3, 32, 32) and images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert loaders["train"].drop_last is True
    assert len(loaders["train"]) == 3  # 12 images / batch 4
    assert loaders["val"].drop_last is False
    assert sum(len(b[1]) for b in loaders["test"]) == 3


def test_small_train_split_still_yields_a_batch(dataset_dir):
    loaders = build_dataloaders(config(*dataset_dir, batch_size=64), ("train",), device="cpu")

    assert loaders["train"].drop_last is False
    batches = list(loaders["train"])
    assert len(batches) == 1 and len(batches[0][1]) == 12


def test_eval_loaders_keep_manifest_order(dataset_dir):
    loader = build_dataloaders(config(*dataset_dir), ("test_official",), device="cpu")[
        "test_official"
    ]
    labels = torch.cat([b[1] for b in loader]).tolist()
    assert labels == loader.dataset.targets


def test_training_shuffle_is_seeded(dataset_dir):
    def epoch_order(seed: int) -> list[list[int]]:
        loader = build_dataloaders(config(*dataset_dir, seed=seed), ("train",), device="cpu")
        loader = loader["train"]
        return [list(loader.batch_sampler.sampler) for _ in range(2)]

    first, again, other = epoch_order(42), epoch_order(42), epoch_order(7)
    assert first == again
    assert first[0] != first[1]  # each epoch reshuffles
    assert first != other


def test_pin_memory_only_on_cuda(dataset_dir):
    cpu = build_dataloaders(config(*dataset_dir), ("val",), device="cpu")["val"]
    cuda = build_dataloaders(config(*dataset_dir), ("val",), device="cuda")["val"]
    off = build_dataloaders(config(*dataset_dir, pin_memory=False), ("val",), device="cuda")

    assert cpu.pin_memory is False
    assert cuda.pin_memory is True  # configured only; no GPU is used by this test
    assert off["val"].pin_memory is False


def test_workers_are_persistent_when_enabled(dataset_dir):
    loader = build_dataloaders(config(*dataset_dir, num_workers=2), ("val",), device="cpu")["val"]
    assert loader.num_workers == 2 and loader.persistent_workers is True


@pytest.mark.integration
def test_worker_processes_can_load_batches(dataset_dir):
    loader = build_dataloaders(config(*dataset_dir, num_workers=2), ("train",), device="cpu")
    images, labels = next(iter(loader["train"]))
    assert images.shape == (4, 3, 32, 32) and len(labels) == 4


# --- Configuration ------------------------------------------------------------------


def test_project_config_uses_data_yaml_paths(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    data = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())

    cfg = load_data_config()

    assert cfg.manifest == Path(data["split"]["output"])
    assert cfg.image_root == Path(data["archive"]["output"])
    assert cfg.class_names == ("buildings", "forest", "glacier", "mountain", "sea", "street")
    assert (cfg.image_size, cfg.batch_size, cfg.eval_batch_size) == (224, 16, 64)
    assert cfg.augmentation.crop_scale == (0.6, 1.0)
    assert cfg.seed == 42


def test_num_classes_must_match_class_names(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    train = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text())
    train["data"]["num_classes"] = 5
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(train))

    with pytest.raises(ManifestError, match="num_classes is 5"):
        load_data_config(path)
