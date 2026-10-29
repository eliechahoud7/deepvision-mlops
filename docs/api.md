# Inference API

A FastAPI service that classifies scene images into `buildings`, `forest`, `glacier`,
`mountain`, `sea`, or `street` using the trained checkpoint.

## How it works

- The model is loaded **once at start-up** from `artifacts/checkpoints/best.pt` with
  `torch.load(weights_only=True)`, through the same helper as held-out evaluation
  (`deepvision.evaluation.checkpoint`). The network is rebuilt from the checkpoint's
  metadata with `pretrained=False`: nothing is downloaded.
- Images get the **same deterministic preprocessing as validation and test**
  (resize the shorter side to the checkpoint's image size, centre crop, ImageNet
  normalization), so served predictions come from exactly the evaluated model.
- Inference runs in eval mode under `torch.inference_mode()` in **FP32**, on **CPU** by
  default (about 30 ms per image for ResNet-18 on a 4-core laptop CPU).
- Image content decides the format, never the filename or content type. JPEG, PNG,
  WebP, and BMP are accepted; EXIF orientation is applied; images are converted to RGB.
- Uploads are size-limited while they are received, kept in memory, and never stored.

## Running

From the repository root, with the virtual environment active and the checkpoint
present (`dvc repro train` or `dvc pull`):

```powershell
uvicorn deepvision.api.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Interactive OpenAPI documentation: `http://127.0.0.1:8000/docs`.

**Run one worker.** Each Uvicorn worker is a separate process that loads its own copy of
the model, and a single forward pass already uses several CPU threads. Within the worker,
prediction endpoints run in FastAPI's thread pool (the event loop stays free for health
checks) and the predictor serializes forward passes with a lock so concurrent requests
do not oversubscribe the CPU. Scale out with more containers or processes behind a load
balancer later, sized to the available cores.

## Configuration

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `DEEPVISION_CHECKPOINT` | `artifacts/checkpoints/best.pt` | Checkpoint to serve |
| `DEEPVISION_EVALUATION_METRICS` | `reports/evaluation/metrics.json` | Held-out metrics shown by `/model` (only if they belong to this checkpoint) |
| `DEEPVISION_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `DEEPVISION_MAX_UPLOAD_MB` | `10` | Maximum size of one uploaded file |
| `DEEPVISION_MAX_BATCH` | `16` | Maximum files per `/predict/batch` request |
| `DEEPVISION_TOP_K` | `3` | Classes returned in `top_k` |
| `DEEPVISION_MAX_PIXELS` | `40000000` | Maximum width × height, checked before decoding |
| `DEEPVISION_NUM_THREADS` | PyTorch default | CPU threads per forward pass |

Invalid values stop the service at start-up with a clear message.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness: always `200 {"status": "ok"}` while the process runs |
| GET | `/ready` | Readiness: `200` once the model is loaded, otherwise `503` |
| GET | `/model` | Model version, backbone, classes, image size, epoch, checkpoint MD5, training commit, validation metrics, and held-out test metrics |
| POST | `/predict` | Classify one image (multipart field `file`) |
| POST | `/predict/batch` | Classify up to `DEEPVISION_MAX_BATCH` images (multipart field `files`, repeated); results keep request order |
| GET | `/metrics` | Prometheus metrics |

If the checkpoint cannot be loaded, the process still starts: `/health` returns 200 while
`/ready`, `/model`, and the prediction endpoints return 503. Liveness and readiness can
therefore be probed separately.

### Examples

```powershell
curl.exe -F "file=@photo.jpg" http://127.0.0.1:8000/predict
curl.exe -F "files=@a.jpg" -F "files=@b.png" http://127.0.0.1:8000/predict/batch
```

`POST /predict` response:

```json
{
  "label": "mountain",
  "confidence": 0.912344,
  "top_k": [
    {"label": "mountain", "probability": 0.912344},
    {"label": "glacier", "probability": 0.081207},
    {"label": "sea", "probability": 0.004113}
  ],
  "probabilities": {"buildings": 0.000412, "forest": 0.000981, "glacier": 0.081207,
                    "mountain": 0.912344, "sea": 0.004113, "street": 0.000943},
  "model_version": "resnet18-0fce8f2bc9a2",
  "inference_ms": 31.4
}
```

`POST /predict/batch` returns `{"predictions": [...], "model_version": ..., "inference_ms": ...}`,
where each prediction has the fields above plus `index`, its position in the request.

`model_version` is the backbone plus the first 12 characters of the checkpoint's MD5.

## Errors

Every error has the same shape and never contains stack traces, filesystem paths, or
image contents:

```json
{"error": {"code": "unsupported_format", "message": "the image: unsupported image format GIF; ..."}}
```

| Status | Code | Cause |
|---|---|---|
| 400 | `invalid_image` | Empty, corrupt, truncated, or unrecognizable image data |
| 400 | `bad_request` | Malformed multipart request |
| 404 / 405 | `not_found` / `method_not_allowed` | Unknown path or wrong HTTP method |
| 413 | `payload_too_large` | A file or the request body exceeds the upload limit |
| 413 | `image_too_large` | Width × height exceeds `DEEPVISION_MAX_PIXELS` |
| 413 | `batch_too_large` | More files than `DEEPVISION_MAX_BATCH` |
| 415 | `unsupported_format` | Not JPEG, PNG, WebP, or BMP |
| 422 | `invalid_request` | Missing or invalid field (`details` lists the problems) |
| 500 | `internal_error` | Unexpected inference failure |
| 503 | `model_unavailable` | The model is not loaded (the message gives a path-free reason) |

## Metrics

`GET /metrics` exposes, for this process:

| Metric | Type | Labels |
|---|---|---|
| `deepvision_requests_total` | counter | `endpoint`, `status` |
| `deepvision_request_latency_seconds` | histogram | `endpoint` |
| `deepvision_inference_latency_seconds` | histogram | — |
| `deepvision_predictions_total` | counter | `label` |
| `deepvision_prediction_confidence` | histogram | — |
| `deepvision_model_info` | gauge (always 1) | `model_version`, `backbone`, `training_commit` |
| `deepvision_model_ready` | gauge | — |

Unknown paths are counted under `endpoint="other"` to keep label cardinality bounded.
Dashboards and alerting belong to the monitoring milestone.

## Limitations and security

- **No authentication, TLS, or rate limiting.** The service binds to `127.0.0.1` and is
  meant for local use. Before exposing it to a network, put it behind authentication,
  TLS, request rate limits, and a reverse proxy with its own body-size limits.
- **Single process.** Throughput is roughly one CPU-bound forward pass at a time; use
  `/predict/batch` for bulk work.
- **Out-of-distribution images** (not one of the six scene types) still receive one of the
  six labels; there is no rejection class. Low confidence is a hint, not a guarantee.
- **Known confusions** mirror evaluation: mountain ↔ glacier and street ↔ buildings.
- Requests and predictions are logged (endpoint, status, latency, label, confidence);
  image bytes are never logged or stored. Input logging for drift analysis is future work.
