# syntax=docker/dockerfile:1
#
# DeepVision Streamlit frontend: a client of the inference API, non-root, no model inside.
#
# Contains only the deepvision.frontend package and its HTTP/UI dependencies: no PyTorch,
# no checkpoint, no model code. Predictions come from the API at DEEPVISION_API_URL
# (http://api:8000 on the Compose "frontend" network).

ARG PYTHON_IMAGE=python:3.12.10-slim-bookworm

# ---------------------------------------------------------------- build stage
FROM ${PYTHON_IMAGE} AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements-frontend.txt /tmp/requirements-frontend.txt
RUN pip install -r /tmp/requirements-frontend.txt \
    && python -m compileall -q /opt/venv/lib

# -------------------------------------------------------------- runtime stage
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="deepvision-frontend" \
      org.opencontainers.image.description="DeepVision Streamlit frontend (client of the inference API)"

RUN groupadd --system --gid 10001 deepvision \
    && useradd --system --uid 10001 --gid deepvision --no-create-home \
        --home-dir /nonexistent --shell /usr/sbin/nologin deepvision

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
# Streamlit reads .streamlit/config.toml from the working directory (telemetry off,
# error details hidden, upload limit, theme).
COPY .streamlit/config.toml /app/.streamlit/config.toml
COPY src/deepvision/__init__.py /app/deepvision/__init__.py
COPY src/deepvision/frontend/ /app/deepvision/frontend/
RUN /opt/venv/bin/python -m compileall -q /app/deepvision

# HOME=/tmp: Streamlit may write small state files under ~/.streamlit; /tmp is the only
# writable path when the root filesystem is read-only.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    DEEPVISION_API_URL=http://api:8000

USER 10001:10001
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

# 0.0.0.0 inside the container's network namespace; compose.yaml publishes the port on
# the host's 127.0.0.1 only.
CMD ["streamlit", "run", "/app/deepvision/frontend/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.fileWatcherType=none"]
