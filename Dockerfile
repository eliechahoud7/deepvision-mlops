# syntax=docker/dockerfile:1
#
# DeepVision inference service: CPU-only PyTorch, non-root, no model weights inside.
#
# The checkpoint is NOT copied into the image. Mount it read-only at run time:
#   artifacts/checkpoints/best.pt -> /app/artifacts/checkpoints/best.pt
# (compose.yaml does this). Without it the container starts but /ready returns 503
# and the health check reports "unhealthy".

ARG PYTHON_IMAGE=python:3.12.10-slim-bookworm

# ---------------------------------------------------------------- build stage
FROM ${PYTHON_IMAGE} AS build

ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Same torch/torchvision versions as development, but the CPU builds (no CUDA libraries).
RUN pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"

COPY requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install -r /tmp/requirements-serve.txt

# pyproject.toml reads its dependency list from requirements.txt and its readme from
# README.md, so both are needed to build the package; --no-deps keeps the serving set.
COPY pyproject.toml requirements.txt README.md /src/
COPY src /src/src
RUN pip install --no-deps /src \
    && python -m compileall -q /opt/venv/lib

# -------------------------------------------------------------- runtime stage
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="deepvision-api" \
      org.opencontainers.image.description="DeepVision scene classification inference API (CPU)"

RUN groupadd --system --gid 10001 deepvision \
    && useradd --system --uid 10001 --gid deepvision --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin deepvision

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
# Mount points for the read-only checkpoint and evaluation report.
RUN mkdir -p /app/artifacts/checkpoints /app/reports/evaluation

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEEPVISION_CHECKPOINT=/app/artifacts/checkpoints/best.pt \
    DEEPVISION_EVALUATION_METRICS=/app/reports/evaluation/metrics.json \
    DEEPVISION_DEVICE=cpu

USER 10001:10001
EXPOSE 8000

# Docker has a single health status, so it tracks readiness: healthy once the model is
# loaded. /health (liveness) stays available for orchestrators that probe it separately.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"]

# 0.0.0.0 inside the container's own network namespace; compose.yaml publishes the port
# on the host's 127.0.0.1 only. One worker: each worker would load its own model copy.
CMD ["uvicorn", "deepvision.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--timeout-graceful-shutdown", "10"]
