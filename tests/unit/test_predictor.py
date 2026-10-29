import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import CLASSES, image_bytes
from PIL import Image
from torch.utils.data import DataLoader

from deepvision.data.dataset import ManifestImageDataset, load_image
from deepvision.data.transforms import build_transforms
from deepvision.evaluation import evaluate
from deepvision.inference.predictor import (
    ImageTooLargeError,
    InvalidImageError,
    Predictor,
    UnsupportedFormatError,
    decode_image,
    file_md5,
)
from deepvision.training.engine import TrainingError

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- Decoding -----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "BMP"])
def test_supported_formats_decode_to_rgb(fmt):
    image = decode_image(image_bytes(fmt, size=(40, 30)))
    assert image.mode == "RGB" and image.size == (40, 30)


@pytest.mark.parametrize(
    ("fmt", "mode"), [("PNG", "L"), ("PNG", "RGBA"), ("PNG", "P"), ("PNG", "LA"), ("JPEG", "CMYK")]
)
def test_other_color_modes_are_converted_to_rgb(fmt, mode):
    image = decode_image(image_bytes(fmt, mode=mode))
    assert image.mode == "RGB" and image.size == (40, 30)


@pytest.mark.parametrize("fmt", ["GIF", "TIFF", "ICO"])
def test_unsupported_formats_are_rejected(fmt):
    with pytest.raises(UnsupportedFormatError, match=f"unsupported image format {fmt}"):
        decode_image(image_bytes(fmt, size=(32, 32)))


@pytest.mark.parametrize(
    "data",
    [b"", b"definitely not an image", image_bytes("PNG", size=(64, 64))[:200]],
    ids=["empty", "garbage", "truncated"],
)
def test_malformed_images_are_rejected(data):
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_pixel_limit_is_checked_before_decoding(monkeypatch):
    with pytest.raises(ImageTooLargeError, match="1200 pixels; the limit is 1000"):
        decode_image(image_bytes(size=(40, 30)), max_pixels=1000)

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # Pillow's own bomb guard
    with pytest.raises(ImageTooLargeError, match="decompression limit"):
        decode_image(image_bytes(size=(40, 30)))


def test_exif_orientation_is_applied():
    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    pixels[:, :20] = (255, 0, 0)  # left half red
    pixels[:, 20:] = (0, 0, 255)  # right half blue
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=95, exif=exif.tobytes())

    image = decode_image(buffer.getvalue())

    assert image.size == (20, 40)  # now portrait
    top, bottom = image.getpixel((10, 5)), image.getpixel((10, 35))
    assert top[0] > 200 and top[2] < 60  # the left (red) half is now on top
    assert bottom[2] > 200 and bottom[0] < 60


# --- Predictor ----------------------------------------------------------------------


@pytest.fixture
def predictor(checkpoint) -> Predictor:
    return Predictor(checkpoint, device="cpu", top_k=3, max_batch=16)


def test_preprocessing_matches_the_evaluation_transform(predictor, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(image_bytes("PNG", size=(50, 35), seed=3))

    served = predictor.preprocess(decode_image(path.read_bytes()))
    evaluated = build_transforms("test", 32)(load_image(path))

    assert served.shape == (3, 32, 32) and torch.equal(served, evaluated)


def test_predictions_match_the_evaluation_pipeline(predictor, checkpoint, tmp_path):
    rows = []
    for i, label in enumerate(CLASSES):
        relpath = f"test/{label}/{i}.png"
        (tmp_path / "images" / relpath).parent.mkdir(parents=True)
        (tmp_path / "images" / relpath).write_bytes(image_bytes("PNG", seed=i))
        rows.append({"relpath": relpath, "label": label, "split": "test", "group_id": relpath})
    dataset = ManifestImageDataset(
        rows, tmp_path / "images", CLASSES, "test", transform=build_transforms("test", 32)
    )
    model, _ = evaluate.load_model(checkpoint, tuple(CLASSES), torch.device("cpu"))
    logits, _ = evaluate.predict(model, DataLoader(dataset, batch_size=4), torch.device("cpu"))
    expected = torch.softmax(logits, dim=1)

    images = [decode_image((tmp_path / "images" / r["relpath"]).read_bytes()) for r in rows]
    predictions, _ = predictor.predict(images)

    for prediction, probabilities in zip(predictions, expected, strict=True):
        served = torch.tensor([prediction.probabilities[name] for name in CLASSES])
        assert torch.allclose(served, probabilities, atol=1e-6)


def test_prediction_fields(predictor):
    [prediction], elapsed_ms = predictor.predict([decode_image(image_bytes())])

    assert list(prediction.probabilities) == CLASSES
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert len(prediction.top_k) == 3
    scores = [score for _, score in prediction.top_k]
    assert scores == sorted(scores, reverse=True)
    assert prediction.label == prediction.top_k[0][0]
    assert prediction.confidence == max(prediction.probabilities.values())
    assert elapsed_ms > 0


def test_batches_preserve_order_and_match_single_predictions(checkpoint):
    images = [decode_image(image_bytes(seed=i, size=(30 + i, 30))) for i in range(5)]
    chunked = Predictor(checkpoint, max_batch=2)  # 5 images -> 3 forward passes

    batch, _ = chunked.predict(images)
    singles = [chunked.predict([image])[0][0] for image in images]

    for together, alone in zip(batch, singles, strict=True):
        assert together.label == alone.label
        for name in CLASSES:
            assert together.probabilities[name] == pytest.approx(
                alone.probabilities[name], abs=1e-6
            )
    assert chunked.predict([]) == ([], 0.0)


def test_top_k_is_capped_and_arguments_are_checked(checkpoint):
    assert Predictor(checkpoint, top_k=50).top_k == 6
    with pytest.raises(ValueError):
        Predictor(checkpoint, max_batch=0)


def test_model_info_and_matching_evaluation_metrics(checkpoint, tmp_path):
    md5 = file_md5(checkpoint)
    report = tmp_path / "metrics.json"
    test_metrics = {"images": 2997, "accuracy": 0.9, "macro_f1": 0.91, "ece": 0.02}
    report.write_text(json.dumps({"checkpoint": {"md5": md5}, "test": test_metrics}))

    info = Predictor(checkpoint, evaluation_metrics=report).info

    assert info["model_version"] == f"mobilenet_v3_small-{md5[:12]}"
    assert info["epoch"] == 3 and info["training_commit"] == "a" * 40
    assert info["validation_metrics"]["macro_f1"] == 0.8
    assert "confusion_matrix" not in info["validation_metrics"]
    assert info["evaluation_metrics"] == {
        "test": {"images": 2997, "accuracy": 0.9, "macro_f1": 0.91}
    }


@pytest.mark.parametrize(
    "content", [json.dumps({"checkpoint": {"md5": "stale"}, "test": {}}), "{not json", None]
)
def test_stale_or_missing_evaluation_reports_are_ignored(checkpoint, tmp_path, content):
    report = tmp_path / "metrics.json"
    if content is not None:
        report.write_text(content)
    assert Predictor(checkpoint, evaluation_metrics=report).info["evaluation_metrics"] is None


def test_cuda_request_without_cuda_fails(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(TrainingError, match="CUDA is not available"):
        Predictor(checkpoint, device="cuda")


@pytest.mark.integration
def test_real_checkpoint_matches_evaluation_code_on_unlabelled_images():
    checkpoint = REPO_ROOT / "artifacts" / "checkpoints" / "best.pt"
    images_dir = REPO_ROOT / "data" / "interim" / "intel-image-classification" / "pred"
    if not checkpoint.is_file() or not images_dir.is_dir():
        pytest.skip("the trained checkpoint or extracted dataset is not available")
    paths = sorted(images_dir.glob("*.jpg"))[:8]  # unlabelled serving stream, not the test set

    predictor = Predictor(
        checkpoint, evaluation_metrics=REPO_ROOT / "reports/evaluation/metrics.json"
    )
    predictions, _ = predictor.predict([decode_image(p.read_bytes()) for p in paths])

    model, ck = evaluate.load_model(checkpoint, tuple(predictor.class_names), torch.device("cpu"))
    transform = build_transforms("test", ck["image_size"])
    with torch.inference_mode():
        logits = model(torch.stack([transform(load_image(p)) for p in paths]))
    expected = torch.softmax(logits, dim=1)
    for prediction, probabilities in zip(predictions, expected, strict=True):
        served = torch.tensor([prediction.probabilities[c] for c in predictor.class_names])
        assert torch.allclose(served, probabilities, atol=1e-5)
    assert predictor.info["evaluation_metrics"]["test"]["images"] == 2997
