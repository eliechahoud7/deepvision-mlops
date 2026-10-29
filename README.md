# DeepVision

DeepVision is an end-to-end computer vision and MLOps portfolio project: a six-class
image classifier built with PyTorch transfer learning, wrapped in the tooling needed to
train, track, serve, and monitor it.

> **Status:** the data pipeline, training, held-out evaluation, a local FastAPI
> inference service, its Docker image, GitHub Actions CI, and Prometheus/Grafana
> monitoring are implemented. Evidently drift analysis is not implemented yet.

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

# 2. Build and start (first build downloads ~200 MB of CPU PyTorch wheels).
#    This also starts Prometheus and Grafana (see Monitoring); for the API alone use
#    `docker compose up --build -d api`.
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

## Monitoring

Prometheus scrapes the API's `/metrics` endpoint and Grafana shows a provisioned
dashboard. Both run locally from `compose.yaml` with pinned, free images: no accounts
and no cloud services. Grafana's telemetry, update checks and plugin downloads are disabled.

```
                    Compose network "inference"          Compose network "monitoring"
 ┌───────────────┐  scrape /metrics every 15 s  ┌────────────┐  PromQL  ┌─────────┐
 │ api :8000     │ ◄─────────────────────────── │ prometheus │ ◄─────── │ grafana │
 │ FastAPI + CPU │                              │ :9090      │          │ :3000   │
 │ PyTorch       │                              │ rules,     │          │ dash-   │
 └───────┬───────┘                              │ 15-day TSDB│          │ board   │
         │                                      └─────┬──────┘          └────┬────┘
   127.0.0.1:8000                              127.0.0.1:9090           127.0.0.1:3000
```

Grafana is on a separate network from the API and can only reach Prometheus.

**Start it** (Windows PowerShell, repository root; the checkpoint must exist as for Docker):

```powershell
$env:GRAFANA_ADMIN_PASSWORD = "choose-a-password"   # optional; defaults to "admin"
docker compose up --build -d
docker compose ps        # api, prometheus and grafana all show "(healthy)"

Start-Process http://127.0.0.1:3000/d/deepvision-api   # Grafana dashboard (no login to view)
Start-Process http://127.0.0.1:9090/alerts             # Prometheus alerts and rules

docker compose down      # stop; metric history is kept in Docker volumes
docker compose down -v   # stop and delete the Prometheus and Grafana data
```

Ports can be changed with `$env:PROMETHEUS_PORT` and `$env:GRAFANA_PORT`, as with
`$env:DEEPVISION_PORT`. They are always bound to `127.0.0.1`.

**Metrics.** The API exports its own metrics from a private registry
(see `docs/api.md`): request counts by endpoint and HTTP status, request latency,
requests in progress, model inference latency, predictions by class, prediction
confidence, model version, model readiness, and process CPU/memory. Unexpected server
errors are also counted, as JSON `500` responses.

**Dashboard: "DeepVision API"** (folder *DeepVision*, provisioned from
`monitoring/grafana/dashboards/deepvision-api.json`):

| Section | Panels |
|---|---|
| Overview | Model availability, API scrape status, model version, request rate, 5xx error ratio, p95 `/predict` latency |
| Traffic and errors | Requests per second by endpoint, responses by status code, 4xx/5xx errors by endpoint, 5xx ratio |
| Latency | HTTP p50/p95/p99 for `/predict` and `/predict/batch`, model inference p50/p95, average latency by endpoint, in-flight requests |
| Model | Model availability over time, predictions per second by class, prediction confidence (p10 / median) |
| API process | Resident memory and CPU usage |

Traffic panels exclude `/metrics`, `/health`, and `/ready`, which are called continuously
by Prometheus and the Docker health check. Anonymous visitors can view dashboards but not
edit them. Changes made in the UI aren't saved: edit the JSON file instead (it is
reloaded within a minute).

**Alerts** (`monitoring/prometheus/rules/deepvision.yml`, shown at
`http://127.0.0.1:9090/alerts`; no Alertmanager, so nothing is sent anywhere):

| Alert | Fires when |
|---|---|
| `DeepVisionApiDown` | Prometheus cannot scrape the API for 1 minute |
| `DeepVisionModelNotReady` | `deepvision_model_ready == 0` for 2 minutes |
| `DeepVisionHighErrorRate` | More than 5% of API requests return 5xx over 5 minutes, for 5 minutes |
| `DeepVisionHighLatency` | `/predict` p95 latency above 1 s for 10 minutes |

The rules have unit tests (`monitoring/prometheus/tests/`), run in CI with `promtool`.
To run them locally:

```powershell
docker run --rm -v "${PWD}\monitoring\prometheus:/etc/prometheus:ro" -w /etc/prometheus `
  --entrypoint promtool prom/prometheus:v3.15.1 test rules tests/deepvision_rules_test.yml
```

**Security.** Prometheus and Grafana get the API's hardening: localhost-only ports,
non-root users, read-only root filesystems, read-only configuration mounts, all
capabilities dropped, `no-new-privileges`, and memory/PID/log limits. Prometheus's
admin and lifecycle APIs stay disabled, and its storage is capped at 15 days or 1 GB.
Grafana's admin password comes from `GRAFANA_ADMIN_PASSWORD`. The `admin` fallback is
only acceptable because Grafana is bound to localhost; set your own in a `.env` file,
which Git ignores.
