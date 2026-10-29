"""FastAPI inference service for the DeepVision scene classifier.

Launch locally (binds to 127.0.0.1 only):

    uvicorn deepvision.api.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1

The model is loaded once at start-up. If loading fails the process keeps running:
/health (liveness) stays 200 while /ready and the prediction endpoints return 503, so
an orchestrator can tell "process alive" from "model usable".

Run a single Uvicorn worker: each worker process would load its own copy of the model,
and inference already uses several CPU threads per forward pass. Within the worker,
prediction endpoints are synchronous (run in FastAPI's thread pool, keeping the event
loop free for health checks) and the predictor serializes forward passes with a lock.

Uploads are size-limited while they are received and kept in memory (never written to
disk or stored). Error responses and logs never include stack traces, filesystem
paths, or image contents. There is no authentication: add authentication, TLS, and
rate limiting before exposing the service beyond localhost (see docs/api.md).
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.formparsers import MultiPartParser
from starlette.requests import Request

from deepvision.api.schemas import (
    BatchItem,
    BatchPredictionResponse,
    ClassProbability,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadyResponse,
)
from deepvision.api.settings import Settings
from deepvision.evaluation.checkpoint import CheckpointError
from deepvision.inference.predictor import ImageError, Prediction, Predictor
from deepvision.training.engine import TrainingError

logger = logging.getLogger("deepvision.api")

FORM_OVERHEAD_BYTES = 64 * 1024  # multipart boundaries and headers around the files
ENDPOINTS = frozenset({"/health", "/ready", "/model", "/predict", "/predict/batch", "/metrics"})
LOAD_FAILURES = {
    "not_found": "the model checkpoint is not available",
    "unreadable": "the model checkpoint could not be read",
    "invalid": "the model checkpoint is invalid",
    "unsupported_format": "the model checkpoint format is not supported",
    "unsupported_normalization": "the model checkpoint uses unsupported preprocessing",
    "device_unavailable": "the configured inference device is not available",
    "load_failed": "the model could not be loaded",
}
STATUS_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed"}
ERROR_RESPONSES = {
    status: {"model": ErrorResponse, "description": description}
    for status, description in {
        400: "Malformed or unreadable image",
        413: "Upload, image, or batch too large",
        415: "Unsupported image format",
        422: "Invalid request (for example, a missing file field)",
        500: "Unexpected server error",
        503: "Model not ready",
    }.items()
}  # fmt: skip


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_response(status: int, code: str, message: str, details=None) -> JSONResponse:
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return JSONResponse(status_code=status, content={"error": error})


class ServiceMetrics:
    """Prometheus metrics in a private registry (one per app instance)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "deepvision_requests", "HTTP requests", ["endpoint", "status"], registry=self.registry
        )
        self.request_latency = Histogram(
            "deepvision_request_latency_seconds",
            "HTTP request latency",
            ["endpoint"],
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "deepvision_inference_latency_seconds",
            "Preprocessing and model time per prediction request",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
            registry=self.registry,
        )
        self.predictions = Counter(
            "deepvision_predictions", "Predicted images by class", ["label"], registry=self.registry
        )
        self.confidence = Histogram(
            "deepvision_prediction_confidence",
            "Confidence of the predicted class",
            buckets=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
            registry=self.registry,
        )
        self.model_info = Gauge(
            "deepvision_model_info",
            "Loaded model (value is always 1)",
            ["model_version", "backbone", "training_commit"],
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "deepvision_model_ready", "1 if the model is loaded", registry=self.registry
        )


class _PayloadTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """Reject request bodies above a per-path limit while they are being received."""

    def __init__(self, app, limits: dict[str, int]) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope, receive, send) -> None:
        limit = self.limits.get(scope.get("path")) if scope["type"] == "http" else None
        if limit is None:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await self._reject(scope, send, limit)
            return

        state = {"received": 0, "exceeded": False, "started": False}

        async def limited_receive():
            message = await receive()
            if message["type"] == "http.request":
                state["received"] += len(message.get("body", b""))
                if state["received"] > limit:
                    state["exceeded"] = True
                    raise _PayloadTooLarge
            return message

        async def guarded_send(message):
            # FastAPI turns errors raised while reading the form into a 400; replace it.
            if state["exceeded"]:
                if message["type"] == "http.response.start" and not state["started"]:
                    state["started"] = True
                    await self._reject(scope, send, limit)
                return
            if message["type"] == "http.response.start":
                state["started"] = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _PayloadTooLarge:
            if not state["started"]:
                await self._reject(scope, send, limit)

    @staticmethod
    async def _reject(scope, send, limit: int) -> None:
        response = error_response(
            413, "payload_too_large", f"the request body exceeds the {limit} byte limit"
        )

        async def no_body():  # the remaining request body is not read
            return {"type": "http.disconnect"}

        await response(scope, no_body, send)


def configure_logging() -> None:
    """Send service logs to stderr unless logging is already configured (Uvicorn only
    configures its own loggers, so deepvision.api messages would otherwise be dropped)."""
    if logger.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def create_app(settings: Settings | None = None, *, predictor_factory=Predictor) -> FastAPI:
    """Build the application; pass settings to override DEEPVISION_* environment variables."""
    configure_logging()
    settings = settings or Settings.from_env()
    metrics = ServiceMetrics()
    single_limit = settings.max_upload_bytes + FORM_OVERHEAD_BYTES
    batch_limit = settings.max_upload_bytes * settings.max_batch + FORM_OVERHEAD_BYTES
    # Keep accepted uploads in memory: Starlette spools larger parts to temporary files.
    MultiPartParser.spool_max_size = max(MultiPartParser.spool_max_size, batch_limit)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor, app.state.load_error = None, None
        try:
            predictor = predictor_factory(
                settings.checkpoint,
                device=settings.device,
                top_k=settings.top_k,
                max_batch=settings.max_batch,
                max_pixels=settings.max_pixels,
                num_threads=settings.num_threads,
                evaluation_metrics=settings.evaluation_metrics,
            )
        except CheckpointError as exc:
            app.state.load_error = exc.code
        except TrainingError:
            app.state.load_error = "device_unavailable"
        except Exception as exc:  # never crash the process: report it through /ready
            app.state.load_error = "load_failed"
            logger.error("model load failed with %s", type(exc).__name__)
        else:
            app.state.predictor = predictor
            metrics.model_ready.set(1)
            metrics.model_info.labels(
                predictor.version, predictor.info["backbone"], predictor.info["training_commit"]
            ).set(1)
            logger.info("model %s loaded on %s", predictor.version, predictor.device)
        if app.state.load_error:
            metrics.model_ready.set(0)
            logger.error("model not ready: %s", app.state.load_error)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="DeepVision inference API",
        description="Scene classification (buildings, forest, glacier, mountain, sea, street).",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.metrics = metrics
    app.add_middleware(
        BodyLimitMiddleware, limits={"/predict": single_limit, "/predict/batch": batch_limit}
    )

    @app.middleware("http")
    async def observe_requests(request: Request, call_next):
        endpoint = request.url.path if request.url.path in ENDPOINTS else "other"
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        metrics.requests.labels(endpoint, str(response.status_code)).inc()
        metrics.request_latency.labels(endpoint).observe(elapsed)
        logger.info(
            "%s %s -> %d in %.1f ms", request.method, endpoint, response.status_code, elapsed * 1000
        )
        return response

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        details = [
            {"loc": list(err.get("loc", ())), "msg": err.get("msg", "")} for err in exc.errors()
        ]
        return error_response(422, "invalid_request", "the request is invalid", details)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        code = STATUS_CODES.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else "request failed"
        return error_response(exc.status_code, code, message)

    def require_predictor() -> Predictor:
        predictor = app.state.predictor
        if predictor is None:
            reason = LOAD_FAILURES.get(app.state.load_error, LOAD_FAILURES["load_failed"])
            raise ApiError(503, "model_unavailable", f"the model is not ready: {reason}")
        return predictor

    def read_image(upload: UploadFile, predictor: Predictor, position: str):
        data = upload.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise ApiError(
                413,
                "payload_too_large",
                f"{position} exceeds the {settings.max_upload_bytes} byte upload limit",
            )
        try:
            return predictor.decode(data)
        except ImageError as exc:
            status = {"unsupported_format": 415, "image_too_large": 413}.get(exc.code, 400)
            raise ApiError(status, exc.code, f"{position}: {exc}") from None

    def run_inference(predictor: Predictor, images) -> tuple[list[Prediction], float]:
        try:
            predictions, elapsed_ms = predictor.predict(images)
        except Exception as exc:
            logger.error("inference failed with %s", type(exc).__name__)
            raise ApiError(500, "internal_error", "inference failed") from None
        metrics.inference_latency.observe(elapsed_ms / 1000)
        for prediction in predictions:
            metrics.predictions.labels(prediction.label).inc()
            metrics.confidence.observe(prediction.confidence)
        return predictions, elapsed_ms

    def result_fields(prediction: Prediction) -> dict:
        return {
            "label": prediction.label,
            "confidence": round(prediction.confidence, 6),
            "top_k": [
                ClassProbability(label=label, probability=round(probability, 6))
                for label, probability in prediction.top_k
            ],
            "probabilities": {k: round(v, 6) for k, v in prediction.probabilities.items()},
        }

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        """Liveness: the process is running (independent of the model)."""
        return HealthResponse(status="ok")

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        responses={503: ERROR_RESPONSES[503]},
        tags=["service"],
    )
    def ready() -> ReadyResponse:
        """Readiness: the model is loaded and predictions can be served."""
        predictor = require_predictor()
        return ReadyResponse(status="ready", model_version=predictor.version)

    @app.get(
        "/model",
        response_model=ModelInfoResponse,
        responses={503: ERROR_RESPONSES[503]},
        tags=["model"],
    )
    def model_info() -> ModelInfoResponse:
        """Metadata of the loaded checkpoint and its validation and held-out test metrics."""
        return ModelInfoResponse(**require_predictor().info)

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses=ERROR_RESPONSES,
        tags=["prediction"],
    )
    def predict(
        file: Annotated[UploadFile, File(description="JPEG, PNG, WebP, or BMP image")],
    ) -> PredictionResponse:
        """Classify one image."""
        predictor = require_predictor()
        image = read_image(file, predictor, "the image")
        [prediction], elapsed_ms = run_inference(predictor, [image])
        logger.info("predicted %s (%.3f)", prediction.label, prediction.confidence)
        return PredictionResponse(
            **result_fields(prediction),
            model_version=predictor.version,
            inference_ms=round(elapsed_ms, 3),
        )

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses=ERROR_RESPONSES,
        tags=["prediction"],
    )
    def predict_batch(
        files: Annotated[list[UploadFile], File(description=f"Up to {settings.max_batch} images")],
    ) -> BatchPredictionResponse:
        """Classify several images; results are returned in request order."""
        predictor = require_predictor()
        if len(files) > settings.max_batch:
            raise ApiError(
                413,
                "batch_too_large",
                f"{len(files)} files sent; the batch limit is {settings.max_batch}",
            )
        images = [read_image(f, predictor, f"file {i}") for i, f in enumerate(files)]
        predictions, elapsed_ms = run_inference(predictor, images)
        logger.info("predicted a batch of %d images", len(predictions))
        return BatchPredictionResponse(
            predictions=[BatchItem(index=i, **result_fields(p)) for i, p in enumerate(predictions)],
            model_version=predictor.version,
            inference_ms=round(elapsed_ms, 3),
        )

    @app.get("/metrics", tags=["service"], response_class=Response)
    def prometheus_metrics() -> Response:
        """Prometheus metrics for this process."""
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
