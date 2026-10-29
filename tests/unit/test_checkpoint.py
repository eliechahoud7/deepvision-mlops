import pytest
import torch
import torchvision
from conftest import CLASSES, make_checkpoint

from deepvision.evaluation.checkpoint import CheckpointError, load_trained_model


def test_loads_an_eval_mode_model_without_downloading(checkpoint, monkeypatch):
    requested = []
    real = torchvision.models.get_model

    def spy(name, weights=None):
        requested.append(weights)
        return real(name, weights=weights)

    monkeypatch.setattr(torchvision.models, "get_model", spy)
    model, loaded = load_trained_model(checkpoint, class_names=CLASSES)

    assert requested == [None]
    assert not model.training and loaded["backbone"] == "mobilenet_v3_small"
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 6)


@pytest.mark.parametrize(
    ("setup", "code", "message"),
    [
        (lambda p: None, "not_found", "dvc repro train"),
        (lambda p: p.write_bytes(b"not a checkpoint"), "unreadable", "cannot read"),
        (lambda p: torch.save({"format_version": 1}, p), "invalid", "missing required fields"),
        (lambda p: make_checkpoint(p, format_version=99), "unsupported_format", "format 99"),
        (
            lambda p: make_checkpoint(p, normalization={"mean": [0.5] * 3, "std": [0.5] * 3}),
            "unsupported_normalization",
            "not ImageNet's",
        ),
        (lambda p: make_checkpoint(p, backbone="resnet18"), "invalid", "do not fit"),
    ],
)
def test_bad_checkpoints_raise_coded_errors(tmp_path, setup, code, message):
    path = tmp_path / "best.pt"
    setup(path)
    with pytest.raises(CheckpointError, match=message) as info:
        load_trained_model(path)
    assert info.value.code == code


def test_class_names_must_match_when_given(checkpoint):
    load_trained_model(checkpoint)  # no expectation: any classes accepted
    with pytest.raises(CheckpointError, match="differ from the configured class_names") as info:
        load_trained_model(checkpoint, class_names=list(reversed(CLASSES)))
    assert info.value.code == "class_mismatch"
