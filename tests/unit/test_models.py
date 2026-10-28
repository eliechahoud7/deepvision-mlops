import pytest
import torch
import torchvision
from torch import nn

from deepvision.models.factory import SUPPORTED_BACKBONES, ImageClassifier, build_model

BATCHNORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def batchnorms(modules) -> list[nn.Module]:
    return [m for module in modules for m in module.modules() if isinstance(m, BATCHNORM)]


@pytest.fixture
def resnet() -> ImageClassifier:
    torch.manual_seed(0)
    return build_model("resnet18", num_classes=6, dropout=0.2)


@pytest.mark.parametrize("backbone", sorted(SUPPORTED_BACKBONES))
def test_supported_backbones_output_one_logit_per_class(backbone):
    model = build_model(backbone, num_classes=6, dropout=0.3).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert logits.shape == (2, 6)
    final_linear = [m for m in model.head.modules() if isinstance(m, nn.Linear)][-1]
    assert final_linear.out_features == 6
    dropouts = [m for m in model.head.modules() if isinstance(m, nn.Dropout)]
    assert dropouts and all(m.p == 0.3 for m in dropouts)


def test_head_and_backbone_parameters_partition_the_model(resnet):
    head = {id(p) for p in resnet.head_parameters()}
    backbone = {id(p) for p in resnet.backbone_parameters()}

    assert head and backbone and not head & backbone
    assert head | backbone == {id(p) for p in resnet.parameters()}


def test_freezing_leaves_only_the_head_trainable(resnet):
    resnet.freeze_backbone()

    assert resnet.backbone_frozen
    assert all(p.requires_grad for p in resnet.head_parameters())
    assert not any(p.requires_grad for p in resnet.backbone_parameters())

    resnet.unfreeze_backbone()
    assert not resnet.backbone_frozen
    assert all(p.requires_grad for p in resnet.parameters())


def test_frozen_backbone_stays_in_eval_mode_during_training(resnet):
    resnet.freeze_backbone()
    resnet.train()  # what a training loop calls at the start of every epoch

    assert resnet.training and resnet.head.training
    assert all(not m.training for m in resnet.backbone_modules())
    assert all(not bn.training for bn in batchnorms(resnet.backbone_modules()))

    resnet.unfreeze_backbone()
    assert all(bn.training for bn in batchnorms(resnet.backbone_modules()))


def test_frozen_batchnorm_statistics_do_not_change(resnet):
    resnet.freeze_backbone()
    resnet.train()
    before = [bn.running_mean.clone() for bn in batchnorms(resnet.backbone_modules())]

    resnet(torch.randn(4, 3, 64, 64))

    after = [bn.running_mean for bn in batchnorms(resnet.backbone_modules())]
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))

    resnet.unfreeze_backbone()
    resnet(torch.randn(4, 3, 64, 64))
    changed = [bn.running_mean for bn in batchnorms(resnet.backbone_modules())]
    assert not all(torch.equal(a, b) for a, b in zip(before, changed, strict=True))


def test_frozen_backbone_receives_no_gradients(resnet):
    resnet.freeze_backbone()
    resnet.train()
    loss = nn.functional.cross_entropy(resnet(torch.randn(2, 3, 64, 64)), torch.tensor([0, 5]))
    loss.backward()

    assert all(p.grad is None for p in resnet.backbone_parameters())
    assert all(p.grad is not None for p in resnet.head_parameters())


def test_freeze_at_build_time_and_eval_mode():
    model = build_model("mobilenet_v3_small", num_classes=6, freeze_backbone=True)

    assert model.backbone_frozen
    model.eval()
    assert not any(m.training for m in model.modules())
    model.train()
    assert model.head.training
    assert not any(m.training for m in model.backbone_modules())


def test_freeze_while_in_eval_mode_is_applied_on_next_train(resnet):
    resnet.eval()
    resnet.freeze_backbone()
    assert not resnet.training

    resnet.train()
    assert resnet.head.training and not any(m.training for m in resnet.backbone_modules())


def test_pretrained_requests_default_weights_without_downloading(monkeypatch):
    requested = []
    real_get_model = torchvision.models.get_model

    def fake_get_model(name, weights=None):
        requested.append((name, weights))
        return real_get_model(name, weights=None)

    monkeypatch.setattr(torchvision.models, "get_model", fake_get_model)
    build_model("resnet18", num_classes=6, pretrained=True)
    build_model("resnet18", num_classes=6, pretrained=False)

    assert requested == [("resnet18", "DEFAULT"), ("resnet18", None)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backbone": "vgg16"}, "unsupported backbone"),
        ({"num_classes": 1}, "num_classes"),
        ({"dropout": 1.0}, "dropout"),
        ({"dropout": -0.1}, "dropout"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, message):
    args = {"backbone": "resnet18", "num_classes": 6} | kwargs
    with pytest.raises(ValueError, match=message):
        build_model(args.pop("backbone"), args.pop("num_classes"), **args)
