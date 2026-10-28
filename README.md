# DeepVision

DeepVision is an end-to-end computer vision and MLOps portfolio project: a six-class
image classifier built with PyTorch transfer learning, wrapped in the tooling needed to
train, track, serve, and monitor it.

> **Status:** early scaffolding. The directory structure and dependency files exist;
> training, serving, CI/CD, and monitoring are not implemented yet.

## Planned architecture

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
```
