# DeepVision

DeepVision is an end-to-end computer vision and MLOps portfolio project: a six-class
image classifier built with PyTorch transfer learning, wrapped in the tooling needed to
train, track, serve, and monitor it.

> **Status:** the data pipeline, training, held-out evaluation, and a local FastAPI
> inference service are implemented. Docker, CI/CD, Grafana dashboards, and Evidently
> drift analysis are not implemented yet.

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
docker/             Container definitions
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
