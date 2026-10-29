# Data drift analysis

DeepVision uses [Evidently](https://github.com/evidentlyai/evidently) to check whether
the images the model receives still look like the images it was built on, and whether
the model's predictions on them have shifted. It runs locally and offline: Evidently's
usage telemetry is disabled and report generation makes no network requests.

## Data

| Role | Split | Images | Why |
|---|---|---:|---|
| Reference | `val` | 2,098 | Labelled images from the training distribution, not used for weight updates |
| Current | `pred` | 7,301 | The dataset's unlabelled images: the stand-in for live traffic |

The held-out test splits (`test`, `test_official`, `test_excluded`) are never read. The
configuration rejects them, and the code refuses to load them (see `docs/evaluation.md`).

## Features

Evidently compares tables, so each image becomes one row of 19 columns:

| Group | Columns | Detects |
|---|---|---|
| Input (10) | `width`, `height`, `brightness`, `contrast`, `red_mean`, `green_mean`, `blue_mean`, `saturation`, `sharpness` (Laplacian variance), `entropy` (grey-level, bits) | Lighting, camera, colour, compression, focus, or resolution changes. Needs no labels or model. |
| Prediction (9) | `predicted_label`, `confidence`, `prediction_entropy` (0 = certain, 1 = uniform), `prob_<class>` × 6 | The model seeing the data differently, e.g. a different class mix or lower confidence |

Predictions come from the inference service's `Predictor`, so they use exactly the
served checkpoint and preprocessing.

## Method

Evidently's `DataDriftPreset` tests each column and chooses the test from the column type
and sample size. With more than 1,000 reference rows it uses distances: normalized
**Wasserstein** for numeric columns and **Jensen–Shannon** for categorical ones, each
drifting at ≥ 0.1. With smaller samples it uses p-value tests (K-S, chi-squared), drifting
below 0.05. Distances avoid the problem of p-values on large samples, where trivial shifts
come out as "significant".

**Verdict.** A group drifts when at least `drift_share` (default 50%) of its columns
drift, and the dataset drifts when **either** group drifts. A single share over all 19
columns would let the nine closely related prediction columns outvote a clear input
change that the model happens to tolerate. The simulations below show this happening.

## Results (ResNet-18 baseline)

| Run | Input drift | Prediction drift | Verdict |
|---|---|---|---|
| `pred` vs `val` (real) | no — 0/10 | no — 0/9 | **No drift** |
| Simulated: `pred` darkened to 60% brightness | **yes — 7/10** | no — 0/9 | **Drift** |
| Simulated: `pred` Gaussian-blurred (radius 2) | no — 4/10 | **yes — 7/9** | **Drift** |

On the real data every score is far below 0.1: the largest is `height` at 0.055, image
statistics are around 0.02–0.03, and prediction columns around 0.01. The predicted-class
shares differ by at most 0.6 percentage points.

Darkening moves every brightness and colour statistic (Wasserstein 0.8–1.9), but not the
predictions: the model was trained with brightness jitter. Under the older all-columns
rule this was "7/19, no drift". Blurring mostly shows up in the model: mean confidence
falls from 0.96 to 0.83, the share of `sea` predictions more than doubles (16% → 35%),
and `glacier` drops from 18% to 6%.

## Running

From the repository root, with the virtual environment active:

```powershell
# --single-item runs only this stage: plain `dvc repro drift` would first rerun any
# outdated upstream stage, including training.
dvc repro --single-item drift       # or: python -m deepvision.monitoring.drift
dvc metrics show reports/drift/summary.json
Start-Process reports\drift\report.html   # interactive Evidently report

# Demonstrate detection on altered images (writes elsewhere; the tracked report is untouched)
python -m deepvision.monitoring.drift --simulate darken --output-dir tmp\drift-darken
python -m deepvision.monitoring.drift --simulate blur --output-dir tmp\drift-blur

# For automation: exit code 3 when drift is detected (0 otherwise, 1 on errors)
python -m deepvision.monitoring.drift --fail-on-drift
```

The run takes about 4 minutes on the laptop GPU (9,399 images). Settings are in
`configs/drift.yaml`.

## Outputs

| File | Content | Stored in |
|---|---|---|
| `reports/drift/summary.json` | Verdict, per-group and per-column scores, feature means, predicted-class shares, model version | Git (DVC metrics) |
| `reports/drift/report.html` | Interactive Evidently report (~5 MB, self-contained) | DVC cache |
| `data/processed/drift/{reference,current}.csv` | One row of features per image | DVC cache |

## Limitations

- **No live traffic yet.** The API does not record its inputs, so "current" is the
  dataset's `pred` folder, which was collected the same way as the training data. Logging
  per-request features (never the images) from the API would make this a production
  check; the feature code is ready for it.
- **Both sets come from one source**, so the real result is expected to show no drift.
  The simulations show that detection works.
- **Feature-level, not semantic.** The statistics catch photometric and resolution
  changes. New scene types that look statistically similar are only visible through
  prediction drift. An embedding-based comparison would be the next step.
- **Default Evidently thresholds** are used. Tune them once there is real traffic.
- **`pandas < 3`:** Evidently 0.7.x does not work with pandas 3, so `requirements.txt`
  pins pandas 2.x. The project's own code does not use pandas.
