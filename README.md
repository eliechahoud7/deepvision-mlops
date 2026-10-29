# DeepVision

DeepVision is an end-to-end computer vision and MLOps portfolio project: a six-class
image classifier built with PyTorch transfer learning, wrapped in the tooling needed to
train, track, serve, and monitor it.

> **Status:** the data pipeline, training, held-out evaluation, a local FastAPI
> inference service, and its Docker image are implemented. CI/CD, Prometheus/Grafana
> dashboards, and Evidently drift analysis are not implemented yet.

## Architecture

| Concern | Tool |
|---|---|
| Model training | PyTorch + torchvision (pretrained backbone, transfer learning) |
| Experiment tracking | MLflow |
| Data and pipeline versioning | DVC |
| Inference API | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest |
| Operational monitoring | Prometheus + Grafana |
| Data quality and drift | Evidently |

## Repository layout

```
configs/            Training and pipeline configuration
data/               Local datasets (raw, interim, processed); not committed to Git
src/deepvision/     Python package: data, models, training, evaluation,
                    inference, monitoring, utils, api
tests/              Unit, integration, and smoke tests
scripts/            Command-line entry points
Dockerfile          CPU inference image (compose.yaml runs it)
.github/workflows/  CI/CD pipelines
monitoring/         Prometheus and Grafana configuration
docs/               Project documentation
artifacts/          Local training outputs
```

## Development setup

Requires Python 3.12. PyTorch is installed separately so the build matches your hardware
(this project is developed against `torch 2.5.1+cu121` on a 2 GB GPU):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

## Pipeline

The DVC pipeline (`dvc.yaml`) runs `extract → audit → split → train → evaluate`:

```powershell
dvc repro            # reruns only stages whose inputs changed
dvc metrics show     # audit, split, training, and evaluation metrics
```

See `docs/evaluation.md` for the held-out test policy.

## Results (ResNet-18 baseline)

| Split | Images | Macro-F1 | Accuracy |
|---|---:|---:|---:|
| Validation (model selection) | 2,098 | 0.949 | 0.949 |
| Clean test (primary) | 2,997 | 0.940 [95% CI 0.932–0.948] | 0.939 |
| Official test | 3,000 | 0.940 | 0.939 |

The main remaining confusions are mountain ↔ glacier and street ↔ buildings.

## Serving

A local FastAPI service serves the trained checkpoint (`artifacts/checkpoints/best.pt`)
on CPU by default:

```powershell
uvicorn deepvision.api.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
curl.exe -F "file=@image.jpg" http://127.0.0.1:8000/predict
```

Interactive documentation is at `http://127.0.0.1:8000/docs`. See `docs/api.md` for
configuration, endpoints, errors, and limitations. The service has no authentication and
binds to localhost only; do not expose it publicly without adding protections.

## Docker

The image runs the same service on **CPU-only PyTorch** (`torch 2.5.1+cpu`, same version as
development) as a non-root user. It contains the code and serving dependencies only:
**the checkpoint is not baked in**. Compose bind-mounts `artifacts/checkpoints/best.pt`
and `reports/evaluation/metrics.json` read-only, so a retrained model needs a container
restart, not a rebuild.

Requires Docker Desktop (WSL 2 backend) and the checkpoint on disk. Commands are for
Windows PowerShell, run from the repository root:

```powershell
# 1. The checkpoint must exist as a file (produce it with `dvc pull` or `dvc repro`)
Test-Path artifacts\checkpoints\best.pt -PathType Leaf   # must print True

# 2. Build and start (first build downloads ~200 MB of CPU PyTorch wheels)
docker compose up --build -d
docker compose ps            # wait for STATUS "(healthy)": the model is loaded
docker compose logs -f api   # Ctrl+C stops following, not the container

# 3. Use it
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/model
curl.exe -F "file=@image.jpg" http://127.0.0.1:8000/predict
curl.exe -F "files=@a.jpg" -F "files=@b.png" http://127.0.0.1:8000/predict/batch
curl.exe http://127.0.0.1:8000/metrics

# 4. Stop and remove the container
docker compose down
```

Use `curl.exe` (not the `curl` alias) for uploads: Windows PowerShell 5.1's
`Invoke-RestMethod` cannot send multipart forms.

**Configuration.** Set variables in the shell (or in a `.env` file, which Git ignores)
before `docker compose up`:

```powershell
$env:DEEPVISION_PORT = "8080"          # host port (default 8000; always bound to 127.0.0.1)
$env:DEEPVISION_NUM_THREADS = "4"      # CPU threads per forward pass (default: PyTorch's)
$env:DEEPVISION_MAX_BATCH = "16"       # see docs/api.md for the variables
$env:DEEPVISION_MAX_UPLOAD_MB = "10"
docker compose up -d
```

**Without Compose**, the equivalent `docker run`:

```powershell
docker build -t deepvision-api:latest .
docker run -d --name deepvision-api -p 127.0.0.1:8000:8000 `
  --read-only --tmpfs /tmp:size=64m,noexec,nosuid,nodev `
  --cap-drop ALL --security-opt no-new-privileges --init --memory 2g --pids-limit 256 `
  --mount "type=bind,src=$PWD\artifacts\checkpoints\best.pt,dst=/app/artifacts/checkpoints/best.pt,readonly" `
  --mount "type=bind,src=$PWD\reports\evaluation\metrics.json,dst=/app/reports/evaluation/metrics.json,readonly" `
  deepvision-api:latest
```

With `--read-only`, the `--tmpfs /tmp` is required: PyTorch needs a writable temporary
directory at import.

**Health.** The image's `HEALTHCHECK` polls `/ready`, so the container turns *healthy*
once the model is loaded (about 5 s) and *unhealthy* if it cannot be loaded. `/health`
remains a pure liveness probe. A missing or invalid checkpoint does not crash the
container: `/health` returns 200 and the other endpoints return 503 (see `docs/api.md`).

**Security settings** (in `compose.yaml`): port published on `127.0.0.1` only; non-root
user (UID 10001); read-only root filesystem with a small `noexec` tmpfs at `/tmp`; read-only
model mounts; all Linux capabilities dropped; `no-new-privileges`; memory (2 GB), PID, and
log-size limits; one Uvicorn worker. The build context is an allow-list
(`.dockerignore`), so data, checkpoints, MLflow runs, `.git`, and `.venv` never enter the
image. The API still has no authentication or TLS: keep it on localhost.

**Troubleshooting.** If the container is *unhealthy*, run `docker compose logs api`; the
line `model not ready: not_found` means the checkpoint was missing. Docker Desktop
creates a missing bind-mount source as an **empty directory**, so if `best.pt` was absent
at `docker compose up`, check for an empty `artifacts\checkpoints\best.pt` directory and
remove it before restoring the file with `dvc pull`.
