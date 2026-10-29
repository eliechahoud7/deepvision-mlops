"""Data drift analysis of the image stream with Evidently.

    python -m deepvision.monitoring.drift --config configs/drift.yaml

Compares a current image set with a reference set. Evidently works on tables, so each
image becomes one row of features:

    image statistics   width, height, brightness, contrast, red/green/blue means,
                       saturation, sharpness, entropy. They describe the input itself
                       (lighting, camera, compression, resolution) and need no labels.
    model outputs      predicted_label, confidence, prediction_entropy, prob_<class>.
                       They show whether the model sees the data differently.

Predictions come from the inference service's Predictor, so drift is measured on exactly
the served model and preprocessing. Evidently's DataDriftPreset tests every column; it
chooses the test from the column type and sample size (Wasserstein / Jensen-Shannon
distance above 1,000 reference rows, K-S / chi-squared p-values below).

The two groups are judged separately: "input" drift when at least drift_share of the
image statistics drift, "prediction" drift likewise for the model outputs, and dataset
drift when either group drifts. A single share over all columns would let the nine
correlated model outputs outvote a clear input change the model happens to tolerate
(darkening every image moves all brightness and colour statistics but no prediction).

The held-out test splits are never read: reference and current must be train, val, or
pred (default: val as reference, the unlabelled pred images as current).

Outputs:
    <features_dir>/reference.csv, current.csv   per-image features (DVC cache)
    <reports_dir>/report.html                    interactive Evidently report (DVC cache)
    <reports_dir>/summary.json                   dataset drift, per-column scores, feature
                                                 means, predicted-label shares (DVC metrics)

--simulate darken|blur alters the current images before analysis, to show that drift is
detected; it requires --output-dir so the tracked reports are not overwritten.

Evidently's usage telemetry is disabled (DO_NOT_TRACK, EVIDENTLY_DISABLE_TELEMETRY) and
report generation makes no network requests. Exit codes: 0 success (also when drift is
found), 1 error, 3 drift found with --fail-on-drift.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageEnhance, ImageFilter

from deepvision.data.dataset import DatasetImageError, ManifestError, load_data_config, load_image
from deepvision.evaluation.checkpoint import CheckpointError
from deepvision.inference.predictor import Predictor
from deepvision.training.config import ConfigError
from deepvision.training.engine import TrainingError

DRIFT_SPLITS = ("train", "val", "pred")  # never the held-out test splits
DEVICES = frozenset({"auto", "cpu", "cuda"})
CONFIG_KEYS = {
    "checkpoint", "train_config", "reference_split", "current_split", "batch_size", "device",
    "drift_share", "features_dir", "reports_dir",
}  # fmt: skip
IMAGE_FEATURES = (
    "width", "height", "brightness", "contrast", "red_mean", "green_mean", "blue_mean",
    "saturation", "sharpness", "entropy",
)  # fmt: skip
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
SIMULATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "darken": lambda image: ImageEnhance.Brightness(image).enhance(0.6),
    "blur": lambda image: image.filter(ImageFilter.GaussianBlur(radius=2)),
}
DRIFT_EXIT_CODE = 3


class DriftError(RuntimeError):
    """Raised when drift analysis cannot run or its results are inconsistent."""


@dataclass(frozen=True)
class DriftConfig:
    checkpoint: Path
    train_config: Path
    reference_split: str
    current_split: str
    batch_size: int
    device: str
    drift_share: float
    features_dir: Path
    reports_dir: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def load_drift_config(path: Path = Path("configs/drift.yaml")) -> DriftConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), f"{path} must contain a mapping")
    unknown, missing = set(raw) - CONFIG_KEYS, CONFIG_KEYS - set(raw)
    _require(
        not unknown and not missing, f"{path}: unknown {sorted(unknown)}, missing {sorted(missing)}"
    )
    config = DriftConfig(
        checkpoint=Path(raw["checkpoint"]),
        train_config=Path(raw["train_config"]),
        reference_split=raw["reference_split"],
        current_split=raw["current_split"],
        batch_size=raw["batch_size"],
        device=raw["device"],
        drift_share=raw["drift_share"],
        features_dir=Path(raw["features_dir"]),
        reports_dir=Path(raw["reports_dir"]),
    )
    for name in ("reference_split", "current_split"):
        _require(
            getattr(config, name) in DRIFT_SPLITS,
            f"{name} must be one of {list(DRIFT_SPLITS)}; the held-out test splits are not used",
        )
    _require(config.reference_split != config.current_split, "reference and current must differ")
    _require(isinstance(config.batch_size, int) and config.batch_size >= 1, "batch_size >= 1")
    _require(config.device in DEVICES, f"device must be one of {sorted(DEVICES)}")
    _require(
        isinstance(config.drift_share, int | float)
        and not isinstance(config.drift_share, bool)
        and 0 < config.drift_share <= 1,
        "drift_share must be a number in (0, 1]",
    )
    return config


# --- Features -------------------------------------------------------------------------


def image_features(image: Image.Image) -> dict[str, float]:
    """Label-free statistics of one image; intensities are scaled to [0, 1]."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = rgb @ LUMA
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    saturation = np.divide(high - low, high, out=np.zeros_like(high), where=high > 0)
    if min(luma.shape) >= 3:  # variance of the 4-neighbour Laplacian: higher = sharper
        laplacian = (
            luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
            - 4 * luma[1:-1, 1:-1]
        )  # fmt: skip
        sharpness = float(laplacian.var())
    else:
        sharpness = 0.0
    counts = np.bincount(np.clip(np.rint(luma * 255), 0, 255).astype(np.int64).ravel())
    p = counts[counts > 0] / counts.sum()
    return {
        "width": float(image.width),
        "height": float(image.height),
        "brightness": float(luma.mean()),
        "contrast": float(luma.std()),
        "red_mean": float(rgb[..., 0].mean()),
        "green_mean": float(rgb[..., 1].mean()),
        "blue_mean": float(rgb[..., 2].mean()),
        "saturation": float(saturation.mean()),
        "sharpness": sharpness,
        "entropy": float(-(p * np.log2(p)).sum()),  # bits, 0..8
    }


def split_relpaths(manifest: Path, split: str) -> list[str]:
    if split not in DRIFT_SPLITS:
        raise DriftError(f"split {split!r} is not allowed for drift analysis")
    try:
        with Path(manifest).open(newline="", encoding="utf-8") as f:
            relpaths = [row["relpath"] for row in csv.DictReader(f) if row["split"] == split]
    except (OSError, KeyError) as exc:
        raise ManifestError(f"cannot read split manifest {manifest}: {exc!r}") from exc
    if not relpaths:
        raise DriftError(f"split {split!r} has no images in {manifest}")
    return relpaths


def feature_table(
    relpaths: Sequence[str],
    image_root: Path,
    predictor: Predictor,
    *,
    batch_size: int,
    simulate: Callable[[Image.Image], Image.Image] | None = None,
) -> pd.DataFrame:
    """One row per image: image statistics followed by the model's outputs."""
    rows = []
    for start in range(0, len(relpaths), batch_size):
        batch = relpaths[start : start + batch_size]
        images = [load_image(Path(image_root) / relpath) for relpath in batch]
        if simulate is not None:
            images = [simulate(image) for image in images]
        predictions, _ = predictor.predict(images)
        for relpath, image, prediction in zip(batch, images, predictions, strict=True):
            probabilities = list(prediction.probabilities.values())
            entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
            rows.append(
                {
                    "relpath": relpath,
                    **image_features(image),
                    "predicted_label": prediction.label,
                    "confidence": prediction.confidence,
                    # Normalized to [0, 1]: 0 = certain, 1 = uniform over the classes.
                    "prediction_entropy": entropy / math.log(len(probabilities)),
                    **{f"prob_{k}": v for k, v in prediction.probabilities.items()},
                }
            )
    return pd.DataFrame(rows)


def prediction_columns(class_names: Sequence[str]) -> list[str]:
    return [
        "predicted_label", "confidence", "prediction_entropy", *(f"prob_{c}" for c in class_names)
    ]  # fmt: skip


def numeric_columns(class_names: Sequence[str]) -> list[str]:
    return [*IMAGE_FEATURES, *prediction_columns(class_names)[1:]]


# --- Evidently ------------------------------------------------------------------------


def _disable_telemetry() -> None:
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("EVIDENTLY_DISABLE_TELEMETRY", "1")


def run_report(
    reference: pd.DataFrame, current: pd.DataFrame, numeric: Sequence[str], drift_share: float
):
    """Run Evidently's DataDriftPreset; returns the snapshot (save_html, dict)."""
    _disable_telemetry()
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    columns = [*numeric, "predicted_label"]
    definition = DataDefinition(
        numerical_columns=list(numeric), categorical_columns=["predicted_label"]
    )
    # Constant columns (e.g. every reference image is 150 px wide) make Evidently's
    # correlation step divide by a zero standard deviation; that only yields NaN there.
    with np.errstate(divide="ignore", invalid="ignore"):
        return Report([DataDriftPreset(drift_share=drift_share)]).run(
            current_data=Dataset.from_pandas(current[columns], data_definition=definition),
            reference_data=Dataset.from_pandas(reference[columns], data_definition=definition),
        )


def summarize_drift(snapshot: dict, drift_share: float, groups: dict[str, list[str]]) -> dict:
    """Per-column and per-group results from an Evidently snapshot dict.

    groups maps a group name to its columns; together they must cover the tested columns
    exactly. The per-column decisions are checked against Evidently's own drift count.
    """
    columns, evidently_count = {}, None
    for metric in snapshot["metrics"]:
        config, value = metric["config"], metric["value"]
        if config["type"].endswith(":ValueDrift"):
            method, threshold = config["method"], config["threshold"]
            # p-value tests drift below the threshold; distance measures at or above it.
            drifted = value < threshold if "p_value" in method else value >= threshold
            columns[config["column"]] = {
                "method": method,
                "threshold": threshold,
                "score": round(float(value), 6),
                "drifted": bool(drifted),
            }
        elif config["type"].endswith(":DriftedColumnsCount"):
            evidently_count = int(value["count"])
    if not columns or evidently_count is None:
        raise DriftError("the Evidently report has no column drift results")
    drifted = sorted(name for name, result in columns.items() if result["drifted"])
    if len(drifted) != evidently_count:
        raise DriftError(
            f"{len(drifted)} drifted columns found, but Evidently counted {evidently_count}"
        )
    grouped = [name for names in groups.values() for name in names]
    if sorted(grouped) != sorted(columns):
        raise DriftError(f"groups {sorted(grouped)} do not match the tested {sorted(columns)}")
    group_results = {}
    for group, names in groups.items():
        count = sum(columns[name]["drifted"] for name in names)
        group_results[group] = {
            "drift": count / len(names) >= drift_share,
            "drifted_columns": count,
            "columns_tested": len(names),
            "drifted_share": round(count / len(names), 6),
        }
    return {
        "dataset_drift": any(result["drift"] for result in group_results.values()),
        "drifted_columns": len(drifted),
        "columns_tested": len(columns),
        "drifted_share": round(len(drifted) / len(columns), 6),
        "drift_share_threshold": drift_share,
        "groups": group_results,
        "drifted_column_names": drifted,
        "columns": columns,
    }


# --- Pipeline -------------------------------------------------------------------------


def run_drift(config: DriftConfig, *, simulate: str | None = None) -> dict:
    data = load_data_config(config.train_config)
    predictor = Predictor(config.checkpoint, device=config.device, max_batch=config.batch_size)
    if tuple(predictor.class_names) != tuple(data.class_names):
        raise DriftError(
            f"checkpoint classes {predictor.class_names} differ from {list(data.class_names)}"
        )
    tables = {}
    for role, split in (("reference", config.reference_split), ("current", config.current_split)):
        relpaths = split_relpaths(data.manifest, split)
        transform = SIMULATIONS[simulate] if simulate and role == "current" else None
        print(f"{role}: {len(relpaths)} {split} images", file=sys.stderr)
        tables[role] = feature_table(
            relpaths, data.image_root, predictor, batch_size=config.batch_size, simulate=transform
        )

    numeric = numeric_columns(predictor.class_names)
    snapshot = run_report(tables["reference"], tables["current"], numeric, config.drift_share)
    summary = {
        "reference": {"split": config.reference_split, "images": len(tables["reference"])},
        "current": {"split": config.current_split, "images": len(tables["current"])},
        "simulated": simulate,
        "model_version": predictor.version,
        "checkpoint_md5": predictor.info["checkpoint_md5"],
        **summarize_drift(
            snapshot.dict(),
            config.drift_share,
            {
                "input": list(IMAGE_FEATURES),
                "prediction": prediction_columns(predictor.class_names),
            },
        ),
        "feature_means": {
            name: {
                "reference": round(float(tables["reference"][name].mean()), 6),
                "current": round(float(tables["current"][name].mean()), 6),
            }
            for name in numeric
        },
        "predicted_label_share": {
            label: {
                role: round(float((tables[role]["predicted_label"] == label).mean()), 6)
                for role in ("reference", "current")
            }
            for label in predictor.class_names
        },
    }

    config.features_dir.mkdir(parents=True, exist_ok=True)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    for role, table in tables.items():
        table.to_csv(
            config.features_dir / f"{role}.csv",
            index=False,
            float_format="%.6f",
            lineterminator="\n",
        )
    snapshot.save_html(str(config.reports_dir / "report.html"))
    (config.reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def format_summary(summary: dict) -> str:
    verdict = "DRIFT DETECTED" if summary["dataset_drift"] else "no dataset drift"
    lines = [
        f"{summary['current']['split']} ({summary['current']['images']} images) vs "
        f"{summary['reference']['split']} ({summary['reference']['images']} images)"
        + (f", simulated: {summary['simulated']}" if summary["simulated"] else ""),
        f"{verdict} (a group drifts when at least {summary['drift_share_threshold']:.0%} of "
        f"its columns drift; {summary['drifted_columns']}/{summary['columns_tested']} overall)",
    ]
    for group, result in summary["groups"].items():
        lines.append(
            f"  {group + ' drift:':<18} {'yes' if result['drift'] else 'no ':<4}"
            f"{result['drifted_columns']}/{result['columns_tested']} columns"
        )
    for name in summary["drifted_column_names"]:
        result = summary["columns"][name]
        lines.append(f"  {name:<20} {result['method']}: {result['score']:.4f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepvision.monitoring.drift",
        description="Compare a current image set with a reference set using Evidently.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/drift.yaml"))
    parser.add_argument("--device", choices=sorted(DEVICES), help="override the configured device")
    parser.add_argument(
        "--simulate", choices=sorted(SIMULATIONS), help="alter the current images (demonstration)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write features and reports here instead (required with --simulate)",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help=f"exit with code {DRIFT_EXIT_CODE} when dataset drift is detected",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.simulate and not args.output_dir:
        parser.error("--simulate requires --output-dir (the tracked reports are not overwritten)")
    try:
        config = load_drift_config(args.config)
        if args.device:
            config = replace(config, device=args.device)
        if args.output_dir:
            config = replace(
                config, features_dir=args.output_dir / "features", reports_dir=args.output_dir
            )
        summary = run_drift(config, simulate=args.simulate)
    except (
        DriftError, ConfigError, ManifestError, DatasetImageError, CheckpointError, TrainingError
    ) as exc:  # fmt: skip
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_summary(summary))
    print(f"reports written to {config.reports_dir}")
    return DRIFT_EXIT_CODE if args.fail_on_drift and summary["dataset_drift"] else 0


if __name__ == "__main__":
    sys.exit(main())
