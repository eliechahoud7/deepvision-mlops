import dataclasses
import math
from pathlib import Path

import pytest
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepvision.data.dataset import DataConfig
from deepvision.models.factory import build_model
from deepvision.training import engine
from deepvision.training.config import (
    EarlyStoppingConfig,
    FineTuneConfig,
    MlflowConfig,
    ModelConfig,
    TrainConfig,
    TrainingConfig,
)
from deepvision.training.engine import (
    EarlyStopping,
    NonFiniteLossError,
    TrainingError,
    amp_enabled,
    build_optimizer,
    build_scheduler,
    classification_metrics,
    confusion_matrix,
    evaluate,
    fit,
    load_checkpoint,
    phase_for_epoch,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)

CLASSES = ("a", "b", "c")
CPU = torch.device("cpu")


def make_config(**training_overrides) -> TrainConfig:
    default_fine_tune = FineTuneConfig(
        head_epochs=1, head_learning_rate=1e-2, backbone_learning_rate=1e-3
    )
    fine_tune = training_overrides.pop("fine_tune", default_fine_tune)
    stopping = training_overrides.pop(
        "early_stopping", EarlyStoppingConfig("val_macro_f1", "max", patience=5)
    )
    training = TrainingConfig(
        epochs=3,
        batch_size=4,
        gradient_accumulation_steps=1,
        mixed_precision=True,  # must fall back to full precision on CPU
        optimizer="adamw",
        weight_decay=0.0,
        scheduler="cosine",
        device="cpu",
        fine_tune=fine_tune,
        early_stopping=stopping,
    )
    return TrainConfig(
        seed=0,
        data=DataConfig(Path("splits.csv"), Path("images"), CLASSES, image_size=32, batch_size=4),
        model=ModelConfig("mobilenet_v3_small", pretrained=False, dropout=0.0),
        training=dataclasses.replace(training, **training_overrides),
        checkpoint_dir=Path("checkpoints"),
        mlflow=MlflowConfig("sqlite:///mlflow.db", "test"),
    )


def make_model():
    torch.manual_seed(0)
    return build_model("mobilenet_v3_small", len(CLASSES), dropout=0.0)


def make_tensors(n: int, seed: int) -> TensorDataset:
    generator = torch.Generator().manual_seed(seed)
    targets = torch.arange(n) % len(CLASSES)
    images = torch.randn(n, 3, 32, 32, generator=generator)
    images[:, 0] += targets.float()[:, None, None]  # a learnable class signal
    return TensorDataset(images, targets)


def make_loaders(n_train: int = 16, n_val: int = 9, batch_size: int = 4) -> dict[str, DataLoader]:
    return {
        "train": DataLoader(
            make_tensors(n_train, seed=1),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(0),
        ),
        "val": DataLoader(make_tensors(n_val, seed=2), batch_size=batch_size),
    }


class FixedLogits(nn.Module):
    """Returns preset logits for the sample indices it receives as inputs."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", logits)

    def forward(self, indices):
        return self.table[indices.long()]


def history_without_timing(history):
    return [{k: v for k, v in record.items() if k != "epoch_seconds"} for record in history]


# --- Environment ------------------------------------------------------------------


def test_mixed_precision_is_used_only_on_cuda():
    assert amp_enabled(torch.device("cuda"), True) is True
    assert amp_enabled(torch.device("cuda"), False) is False
    assert amp_enabled(CPU, True) is False


def test_cuda_request_without_cuda_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == CPU
    with pytest.raises(TrainingError, match="CUDA is not available"):
        resolve_device("cuda")


# --- Metrics ----------------------------------------------------------------------


def test_confusion_matrix_and_macro_metrics_by_hand():
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 1, 0, 2])
    matrix = confusion_matrix(targets, predictions, 3)

    assert matrix.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    metrics = classification_metrics(matrix)
    # precision: a 1/2, b 2/3, c 1/1; recall: a 1/2, b 2/2, c 1/2
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["macro_precision"] == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert metrics["macro_recall"] == pytest.approx((1 / 2 + 1 + 1 / 2) / 3)
    f1 = [0.5, 2 * (2 / 3) * 1 / (2 / 3 + 1), 2 * 1 * 0.5 / 1.5]
    assert metrics["macro_f1"] == pytest.approx(sum(f1) / 3)


def test_macro_metrics_match_scikit_learn_including_missing_predictions():
    generator = torch.Generator().manual_seed(3)
    targets = torch.randint(0, 5, (200,), generator=generator)
    predictions = torch.randint(0, 4, (200,), generator=generator)  # class 4 never predicted
    metrics = classification_metrics(confusion_matrix(targets, predictions, 5))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=range(5), average="macro", zero_division=0
    )

    assert metrics["macro_precision"] == pytest.approx(precision)
    assert metrics["macro_recall"] == pytest.approx(recall)
    assert metrics["macro_f1"] == pytest.approx(f1)
    assert not any(math.isnan(v) for v in metrics.values())


def test_evaluate_reports_loss_metrics_and_confusion_matrix():
    logits = torch.tensor([[3.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0], [2.0, 0, 0], [0, 0, 4.0]])
    targets = torch.tensor([0, 1, 2, 2, 2])
    loader = DataLoader(TensorDataset(torch.arange(5), targets), batch_size=2)

    result = evaluate(FixedLogits(logits), loader, device=CPU, num_classes=3)

    assert result["loss"] == pytest.approx(nn.functional.cross_entropy(logits, targets).item())
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 2]]
    expected = classification_metrics(torch.tensor(result["confusion_matrix"]))
    assert {k: result[k] for k in expected} == pytest.approx(expected)

    limited = evaluate(FixedLogits(logits), loader, device=CPU, num_classes=3, max_batches=1)
    assert sum(map(sum, limited["confusion_matrix"])) == 2


def test_evaluate_rejects_empty_loaders_and_non_finite_loss():
    empty = DataLoader(TensorDataset(torch.arange(0), torch.arange(0)), batch_size=2)
    with pytest.raises(TrainingError, match="no batches"):
        evaluate(FixedLogits(torch.zeros(1, 3)), empty, device=CPU, num_classes=3)

    nan = FixedLogits(torch.full((2, 3), float("nan")))
    loader = DataLoader(TensorDataset(torch.arange(2), torch.tensor([0, 1])), batch_size=2)
    with pytest.raises(NonFiniteLossError, match="validation"):
        evaluate(nan, loader, device=CPU, num_classes=3)


# --- Train one epoch --------------------------------------------------------------


def _train(model, batches, accumulation_steps, scheduler_factory=None):
    optimizer = torch.optim.SGD(model.head_parameters(), lr=1.0)
    scheduler = scheduler_factory(optimizer) if scheduler_factory else None
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    result = train_one_epoch(
        model,
        batches,
        optimizer,
        device=CPU,
        scaler=scaler,
        accumulation_steps=accumulation_steps,
        scheduler=scheduler,
    )
    return result, scheduler


def test_gradient_accumulation_matches_one_large_batch():
    data = make_tensors(8, seed=4)
    images, targets = data.tensors
    reference, accumulated = make_model(), make_model()
    for model in (reference, accumulated):
        model.freeze_backbone()  # BatchNorm in eval mode, so samples are independent

    _train(reference, [(images, targets)], accumulation_steps=1)
    result, _ = _train(
        accumulated, [(images[:4], targets[:4]), (images[4:], targets[4:])], accumulation_steps=2
    )

    assert result["optimizer_steps"] == 1
    for a, b in zip(reference.head_parameters(), accumulated.head_parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_partial_final_window_is_scaled_by_its_own_size():
    images, targets = make_tensors(12, seed=5).tensors
    windows, reference = make_model(), make_model()
    for model in (windows, reference):
        model.freeze_backbone()
    three = [(images[i : i + 4], targets[i : i + 4]) for i in (0, 4, 8)]
    merged = [(images[:8], targets[:8]), (images[8:], targets[8:])]

    result, scheduler = _train(
        windows,
        three,
        accumulation_steps=2,
        scheduler_factory=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=100),
    )
    _train(reference, merged, accumulation_steps=1)

    assert result["optimizer_steps"] == 2
    assert scheduler.last_epoch == 2  # one scheduler step per optimizer step
    for a, b in zip(windows.head_parameters(), reference.head_parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def test_non_finite_training_loss_is_reported_with_its_position():
    model = make_model()
    images, targets = make_tensors(8, seed=6).tensors
    images[4:] = float("nan")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(NonFiniteLossError, match="epoch 7, batch 2"):
        train_one_epoch(
            model,
            [(images[:4], targets[:4]), (images[4:], targets[4:])],
            optimizer,
            device=CPU,
            scaler=torch.amp.GradScaler("cpu", enabled=False),
            epoch=7,
        )


# --- Optimizer, scheduler, phases -------------------------------------------------


def test_phases_split_epochs():
    assert [phase_for_epoch(e, 2) for e in range(4)] == ["head", "head", "full", "full"]
    assert {phase_for_epoch(e, 0) for e in range(3)} == {"full"}
    assert {phase_for_epoch(e, 3) for e in range(3)} == {"head"}


def test_optimizer_groups_and_learning_rates_per_phase():
    model, training = make_model(), make_config().training

    head = build_optimizer(model, "head", training)
    full = build_optimizer(model, "full", training)

    assert isinstance(full, torch.optim.AdamW)
    assert [(g["name"], g["lr"]) for g in head.param_groups] == [("head", 1e-2)]
    assert [(g["name"], g["lr"]) for g in full.param_groups] == [("backbone", 1e-3), ("head", 1e-2)]
    assert sum(len(g["params"]) for g in full.param_groups) == len(list(model.parameters()))


def test_cosine_schedule_decays_to_zero_and_none_keeps_rate():
    model, training = make_model(), make_config().training
    optimizer = build_optimizer(model, "full", training)
    scheduler = build_scheduler(optimizer, training, total_steps=4)
    rates = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        rates.append([g["lr"] for g in optimizer.param_groups])

    assert rates[1][0] < rates[0][0]
    assert rates[-1] == pytest.approx([0.0, 0.0], abs=1e-12)
    constant = dataclasses.replace(training, scheduler="none")
    assert build_scheduler(optimizer, constant, total_steps=4) is None


# --- Early stopping ---------------------------------------------------------------


def test_early_stopping_max_mode():
    stopper = EarlyStopping("val_macro_f1", "max", patience=2)
    improved = [stopper.update(v, e) for e, v in enumerate([0.5, 0.6, 0.6, 0.55], start=1)]

    assert improved == [True, True, False, False]  # ties are not improvements
    assert (stopper.best, stopper.best_epoch, stopper.should_stop) == (0.6, 2, True)


def test_early_stopping_min_mode_nan_and_state_round_trip():
    stopper = EarlyStopping("val_loss", "min", patience=3)
    assert stopper.update(1.0, 1) and stopper.update(0.8, 2)
    assert not stopper.update(float("nan"), 3)
    assert (stopper.best, stopper.bad_epochs, stopper.should_stop) == (0.8, 1, False)

    restored = EarlyStopping("val_loss", "min", patience=3)
    restored.load_state_dict(stopper.state_dict())
    assert restored == stopper


# --- Fit: phases, freezing, guards ------------------------------------------------


def test_two_phase_fit_freezes_then_fine_tunes(tmp_path):
    model, loaders = make_model(), make_loaders()
    config = make_config(epochs=3, scheduler="none")
    initial_backbone = [p.detach().clone() for p in model.backbone_parameters()]
    initial_head = [p.detach().clone() for p in model.head_parameters()]
    seen = []

    def snapshot(record):
        seen.append(
            {
                "frozen": model.backbone_frozen,
                "backbone": [p.detach().clone() for p in model.backbone_parameters()],
                "head": [p.detach().clone() for p in model.head_parameters()],
            }
        )

    result = fit(model, loaders, config, tmp_path, on_epoch_end=snapshot)

    assert [r["phase"] for r in result.history] == ["head", "full", "full"]
    assert [s["frozen"] for s in seen] == [True, False, False]
    assert all(
        torch.equal(a, b) for a, b in zip(initial_backbone, seen[0]["backbone"], strict=True)
    )
    assert not all(torch.equal(a, b) for a, b in zip(initial_head, seen[0]["head"], strict=True))
    assert not all(
        torch.equal(a, b) for a, b in zip(seen[0]["backbone"], seen[1]["backbone"], strict=True)
    )
    head_record, full_record = result.history[0], result.history[1]
    assert (head_record["lr_head"], head_record["lr_backbone"]) == (1e-2, None)
    assert (full_record["lr_head"], full_record["lr_backbone"]) == (1e-2, 1e-3)
    assert result.completed and not result.stopped_early


def test_fit_records_all_metrics_and_cosine_ends_each_phase_at_zero(tmp_path):
    result = fit(make_model(), make_loaders(), make_config(epochs=3), tmp_path)

    expected = {
        "epoch", "phase", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
        "val_macro_precision", "val_macro_recall", "val_macro_f1", "lr_head", "lr_backbone",
        "epoch_seconds", "improved",
    }  # fmt: skip
    assert set(result.history[0]) == expected
    assert result.history[0]["lr_head"] == pytest.approx(0.0, abs=1e-12)  # end of head phase
    assert result.history[-1]["lr_backbone"] == pytest.approx(0.0, abs=1e-12)
    assert 0 < result.history[1]["lr_backbone"] < 1e-3  # mid full phase


def test_fit_rejects_test_loaders(tmp_path):
    loaders = make_loaders()
    with pytest.raises(ValueError, match="test set must not be used"):
        fit(make_model(), loaders | {"test": loaders["val"]}, make_config(), tmp_path)

    class SplitDataset(TensorDataset):
        split = "test"

    disguised = DataLoader(SplitDataset(*make_tensors(4, seed=3).tensors), batch_size=2)
    with pytest.raises(ValueError, match="'val' loader contains the 'test' split"):
        fit(make_model(), {"train": loaders["train"], "val": disguised}, make_config(), tmp_path)


def test_out_of_memory_is_reported_clearly(tmp_path, monkeypatch):
    model = make_model()

    def out_of_memory(images):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(model, "forward", out_of_memory)
    with pytest.raises(TrainingError, match="out of memory in epoch 1 with batch_size 4") as info:
        fit(model, make_loaders(), make_config(), tmp_path)
    assert isinstance(info.value.__cause__, torch.OutOfMemoryError)
    assert not (tmp_path / "last.pt").exists()


def test_max_batches_limits_each_epoch(tmp_path, monkeypatch):
    seen = []
    original = engine.train_one_epoch

    def counting(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(result["optimizer_steps"])
        return result

    monkeypatch.setattr(engine, "train_one_epoch", counting)
    fit(make_model(), make_loaders(), make_config(epochs=1), tmp_path, max_batches=2)
    assert seen == [2]


# --- Fit: early stopping and checkpoints ------------------------------------------


def scripted_evaluate(values, key="macro_f1"):
    calls = iter(values)

    def fake_evaluate(model, loader, **kwargs):
        value = next(calls)
        metrics = {"loss": 1.0, "accuracy": 0.5, "macro_precision": 0.5, "macro_recall": 0.5}
        metrics |= {"macro_f1": 0.5, key: value, "confusion_matrix": [[1, 0, 0]] * 3}
        return metrics

    return fake_evaluate


def test_fit_stops_early_on_validation_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "evaluate", scripted_evaluate([0.5, 0.7, 0.6, 0.65, 0.9]))
    config = make_config(
        epochs=5, early_stopping=EarlyStoppingConfig("val_macro_f1", "max", patience=2)
    )

    result = fit(make_model(), make_loaders(), config, tmp_path)

    assert [r["epoch"] for r in result.history] == [1, 2, 3, 4]
    assert [r["improved"] for r in result.history] == [True, True, False, False]
    assert (result.best_epoch, result.best_metric) == (2, 0.7)
    assert result.stopped_early and result.completed
    assert load_checkpoint(result.best_checkpoint)["epoch"] == 2
    last = load_checkpoint(result.last_checkpoint)
    assert (last["epoch"], last["completed"], last["stopped_early"]) == (4, True, True)


def test_fit_can_monitor_validation_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "evaluate", scripted_evaluate([1.0, 0.5, 0.6, 0.7], key="loss"))
    config = make_config(epochs=4, early_stopping=EarlyStoppingConfig("val_loss", "min", 2))

    result = fit(make_model(), make_loaders(), config, tmp_path)

    assert (result.best_epoch, result.best_metric, len(result.history)) == (2, 0.5, 4)


def test_checkpoints_are_safe_to_load_and_complete(tmp_path):
    model, config = make_model(), make_config(epochs=2)
    result = fit(model, make_loaders(), config, tmp_path, metadata={"git_commit": "abc123"})

    best = load_checkpoint(result.best_checkpoint)  # weights_only=True
    last = load_checkpoint(result.last_checkpoint)
    for checkpoint in (best, last):
        assert checkpoint["format_version"] == 1
        assert checkpoint["backbone"] == "mobilenet_v3_small"
        assert checkpoint["class_names"] == list(CLASSES)
        assert checkpoint["class_to_index"] == {"a": 0, "b": 1, "c": 2}
        assert checkpoint["num_classes"] == 3 and checkpoint["image_size"] == 32
        assert checkpoint["normalization"]["mean"] == [0.485, 0.456, 0.406]
        assert checkpoint["config"]["data"]["manifest"] == "splits.csv"
        assert checkpoint["metadata"] == {"git_commit": "abc123"}
        assert len(checkpoint["metrics"]["confusion_matrix"]) == 3
    assert "optimizer_state" not in best
    for key in ("optimizer_state", "scheduler_state", "scaler_state", "early_stopping", "rng"):
        assert key in last
    assert len(last["history"]) == 2
    assert not list(tmp_path.glob("*.tmp"))

    restored = build_model("mobilenet_v3_small", 3)
    restored.load_state_dict(last["model_state"])
    images = make_tensors(4, seed=9).tensors[0]
    model.eval(), restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(images), restored(images))


def test_save_checkpoint_replaces_atomically(tmp_path):
    path = tmp_path / "nested" / "x.pt"
    save_checkpoint(path, {"value": 1})
    save_checkpoint(path, {"value": 2})
    assert load_checkpoint(path) == {"value": 2}
    assert [p.name for p in path.parent.iterdir()] == ["x.pt"]


# --- Resume -----------------------------------------------------------------------


def run_fit(tmp_path, *, max_epochs=None, resume=False):
    seed_everything(0 if not resume else 999)  # a resumed run must not depend on fresh seeds
    model, loaders = make_model(), make_loaders()
    config = make_config(epochs=4, fine_tune=FineTuneConfig(2, 1e-2, 1e-3))
    result = fit(
        model,
        loaders,
        config,
        tmp_path,
        max_epochs=max_epochs,
        resume_from=tmp_path / "last.pt" if resume else None,
    )
    return model, result


@pytest.mark.parametrize("interrupt_after", [1, 2, 3])
def test_resume_matches_an_uninterrupted_run(tmp_path, interrupt_after):
    full_model, full = run_fit(tmp_path / "full")

    _, partial = run_fit(tmp_path / "resumed", max_epochs=interrupt_after)
    assert len(partial.history) == interrupt_after and not partial.completed
    resumed_model, resumed = run_fit(tmp_path / "resumed", resume=True)

    assert resumed.completed
    assert history_without_timing(resumed.history) == history_without_timing(full.history)
    assert resumed.best_epoch == full.best_epoch
    for key, value in full_model.state_dict().items():
        assert torch.equal(value, resumed_model.state_dict()[key]), key


def test_resuming_a_completed_run_does_not_train(tmp_path):
    run_fit(tmp_path)
    calls = []
    model, loaders = make_model(), make_loaders()
    config = make_config(epochs=4, fine_tune=FineTuneConfig(2, 1e-2, 1e-3))

    result = fit(
        model,
        loaders,
        config,
        tmp_path,
        resume_from=tmp_path / "last.pt",
        on_epoch_end=calls.append,
    )

    assert result.completed and calls == [] and len(result.history) == 4


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda c: dataclasses.replace(c, training=dataclasses.replace(c.training, epochs=6)),
            "training",
        ),
        (
            lambda c: dataclasses.replace(c, model=dataclasses.replace(c.model, dropout=0.5)),
            "model",
        ),
        (
            lambda c: dataclasses.replace(
                c, data=dataclasses.replace(c.data, class_names=("x", "y", "z"))
            ),
            "class names",
        ),
    ],
)
def test_resume_rejects_a_different_configuration(tmp_path, change, message):
    config = make_config(epochs=2)
    fit(make_model(), make_loaders(), config, tmp_path, max_epochs=1)

    with pytest.raises(TrainingError, match=message):
        fit(
            make_model(), make_loaders(), change(config), tmp_path, resume_from=tmp_path / "last.pt"
        )


def test_resume_rejects_a_different_backbone(tmp_path):
    config = make_config(epochs=2)
    fit(make_model(), make_loaders(), config, tmp_path, max_epochs=1)
    other = build_model("resnet18", 3)

    with pytest.raises(TrainingError, match="backbone"):
        fit(other, make_loaders(), config, tmp_path, resume_from=tmp_path / "last.pt")
