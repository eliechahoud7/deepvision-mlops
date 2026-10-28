"""Evaluation metrics computed from predicted class probabilities.

Builds on the training engine's confusion matrix and macro metrics, adding
per-class scores, top-k accuracy, log loss, expected calibration error, and
bootstrap confidence intervals.
"""

import torch

from deepvision.training.engine import classification_metrics, confusion_matrix


def per_class_metrics(matrix: torch.Tensor, class_names: list[str]) -> list[dict]:
    """Precision, recall, F1, and support per class; undefined ratios count as 0."""
    counts = matrix.double()
    true_positive = counts.diagonal()
    predicted, support = counts.sum(dim=0), counts.sum(dim=1)
    precision = torch.where(predicted > 0, true_positive / predicted.clamp(min=1), 0.0)
    recall = torch.where(support > 0, true_positive / support.clamp(min=1), 0.0)
    total = precision + recall
    f1 = torch.where(total > 0, 2 * precision * recall / total.clamp(min=1e-12), 0.0)
    return [
        {
            "class": name,
            "precision": precision[i].item(),
            "recall": recall[i].item(),
            "f1": f1[i].item(),
            "support": int(support[i].item()),
        }
        for i, name in enumerate(class_names)
    ]


def normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    """Each row divided by its total (recall per true class); empty rows stay zero."""
    counts = matrix.double()
    return counts / counts.sum(dim=1, keepdim=True).clamp(min=1)


def top_k_accuracy(probabilities: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    top = probabilities.topk(min(k, probabilities.shape[1]), dim=1).indices
    return (top == targets[:, None]).any(dim=1).double().mean().item()


def log_loss(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean negative log probability of the true class."""
    true_class = probabilities.double().gather(1, targets[:, None].long()).squeeze(1)
    return -true_class.clamp(min=1e-12).log().mean().item()


def expected_calibration_error(
    probabilities: torch.Tensor, targets: torch.Tensor, bins: int = 15
) -> float:
    """Weighted mean gap between confidence and accuracy over equal-width confidence bins."""
    confidence, predictions = probabilities.double().max(dim=1)
    correct = (predictions == targets).double()
    edges = torch.linspace(0, 1, bins + 1, dtype=torch.float64)[1:-1]
    bin_index = torch.bucketize(confidence, edges)  # bin i covers (edge[i-1], edge[i]]
    error = 0.0
    for b in range(bins):
        in_bin = bin_index == b
        if in_bin.any():
            gap = (correct[in_bin].mean() - confidence[in_bin].mean()).abs().item()
            error += gap * in_bin.double().mean().item()
    return error


def bootstrap_intervals(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
    samples: int,
    seed: int,
    level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap intervals for accuracy and macro-F1 (resampling images)."""
    generator = torch.Generator().manual_seed(seed)
    n = targets.numel()
    values: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
    for _ in range(samples):
        index = torch.randint(0, n, (n,), generator=generator)
        metrics = classification_metrics(
            confusion_matrix(targets[index], predictions[index], num_classes)
        )
        for key in values:
            values[key].append(metrics[key])
    tail = (1 - level) / 2
    intervals = {}
    for key, series in values.items():
        quantiles = torch.tensor(series, dtype=torch.float64).quantile(
            torch.tensor([tail, 1 - tail], dtype=torch.float64)
        )
        intervals[key] = {"low": quantiles[0].item(), "high": quantiles[1].item()}
    return intervals


def summarize(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    class_names: list[str],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict:
    """Headline metrics, per-class metrics, and the confusion matrix for one split."""
    num_classes = len(class_names)
    predictions = probabilities.argmax(dim=1)
    matrix = confusion_matrix(targets, predictions, num_classes)
    headline = {
        "images": targets.numel(),
        **classification_metrics(matrix),
        "top2_accuracy": top_k_accuracy(probabilities, targets, 2),
        "log_loss": log_loss(probabilities, targets),
        "ece": expected_calibration_error(probabilities, targets),
    }
    if bootstrap_samples > 0:
        intervals = bootstrap_intervals(
            targets, predictions, num_classes, bootstrap_samples, bootstrap_seed
        )
        headline["accuracy_ci95"] = intervals["accuracy"]
        headline["macro_f1_ci95"] = intervals["macro_f1"]
    return {
        "headline": headline,
        "per_class": per_class_metrics(matrix, class_names),
        "confusion_matrix": matrix,
    }
