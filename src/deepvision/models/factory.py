"""Image classifiers built from torchvision backbones with a replaced classification head.

The torchvision network is wrapped in ImageClassifier, which knows which submodule
is the head. That lets training freeze the backbone for a head-only phase and
unfreeze it for full fine-tuning. While the backbone is frozen it also stays in
evaluation mode, so BatchNorm running statistics (and any dropout or stochastic
depth inside the backbone) do not change; only the head follows train/eval.
"""

from collections.abc import Iterator

import torchvision
from torch import nn

# Backbone name -> attribute holding its classification head.
SUPPORTED_BACKBONES = {
    "resnet18": "fc",
    "resnet34": "fc",
    "resnet50": "fc",
    "mobilenet_v3_small": "classifier",
    "mobilenet_v3_large": "classifier",
    "efficientnet_b0": "classifier",
}


class ImageClassifier(nn.Module):
    """A torchvision network whose backbone can be frozen independently of its head."""

    def __init__(self, network: nn.Module, backbone: str) -> None:
        super().__init__()
        self.network = network
        self.backbone = backbone
        self.head_name = SUPPORTED_BACKBONES[backbone]
        self.backbone_frozen = False

    def forward(self, images):
        return self.network(images)

    @property
    def head(self) -> nn.Module:
        return getattr(self.network, self.head_name)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        head_ids = {id(p) for p in self.head.parameters()}
        return [p for p in self.network.parameters() if id(p) not in head_ids]

    def backbone_modules(self) -> Iterator[nn.Module]:
        for name, module in self.network.named_children():
            if name != self.head_name:
                yield module

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze the backbone; the head always stays trainable."""
        for parameter in self.backbone_parameters():
            parameter.requires_grad = trainable
        for parameter in self.head_parameters():
            parameter.requires_grad = True
        self.backbone_frozen = not trainable
        self.train(self.training)  # re-apply the mode rules below

    def freeze_backbone(self) -> None:
        self.set_backbone_trainable(False)

    def unfreeze_backbone(self) -> None:
        self.set_backbone_trainable(True)

    def train(self, mode: bool = True) -> "ImageClassifier":
        super().train(mode)
        if mode and self.backbone_frozen:
            for module in self.backbone_modules():
                module.eval()
        return self


def _replace_head(network: nn.Module, backbone: str, num_classes: int, dropout: float) -> None:
    head_name = SUPPORTED_BACKBONES[backbone]
    if head_name == "fc":
        in_features = network.fc.in_features
        network.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return
    # MobileNetV3 and EfficientNet end with [..., Dropout, Linear]; keep the structure.
    classifier = network.classifier
    classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
    for module in classifier:
        if isinstance(module, nn.Dropout):
            module.p = dropout


def build_model(
    backbone: str,
    num_classes: int,
    *,
    pretrained: bool = False,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
) -> ImageClassifier:
    """Build a classifier; pretrained=True loads ImageNet weights (downloaded if not cached)."""
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"unsupported backbone {backbone!r}; expected one of {sorted(SUPPORTED_BACKBONES)}"
        )
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    if not 0 <= dropout < 1:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")

    network = torchvision.models.get_model(backbone, weights="DEFAULT" if pretrained else None)
    _replace_head(network, backbone, num_classes, dropout)
    model = ImageClassifier(network, backbone)
    if freeze_backbone:
        model.freeze_backbone()
    return model
