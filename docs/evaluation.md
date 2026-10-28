# Evaluation policy

DeepVision reports model quality on a held-out test set that is never used to make
modelling decisions.

## Data

| Set | Images | Use |
|---|---:|---|
| `test` (clean) | 2,997 | **Primary result.** The original `seg_test` images minus 3 label conflicts. |
| `test_official` | 3,000 | All original `seg_test` images, reported for comparison with other work. |

The 3 `test_excluded` images are byte-identical to training images that carry a
different label (see `reports/audit/` and `reports/splits/summary.json`). Their
predictions are listed in `reports/evaluation/excluded_predictions.csv`.

## Rules

1. **Model selection uses validation only.** Early stopping, the best checkpoint,
   hyperparameters, and architecture choices are decided on the validation split.
2. **The test set measures a finished model.** Test results must not be used to pick
   between models or settings. A change motivated by validation is retrained and then
   evaluated again; the previous test result is kept in Git history, not overwritten
   silently.
3. **The checkpoint defines the model.** Evaluation rebuilds the network from the
   checkpoint's own metadata with `pretrained=False`, and runs in full precision.
4. **Evaluation checks itself first.** Before any test image is loaded, the validation
   split is re-scored and must match the checkpoint's stored validation metrics within
   `validation_check_tolerance`; otherwise evaluation stops.
5. **Uncertainty is reported.** Accuracy and macro-F1 come with 95% percentile-bootstrap
   confidence intervals over test images.

## Running

```bash
dvc repro evaluate
# or, without DVC bookkeeping:
python -m deepvision.evaluation.evaluate --config configs/evaluate.yaml
```

Outputs are written to `reports/evaluation/` and the test metrics are added to the
checkpoint's MLflow training run as `test_*` and `test_official_*`.
