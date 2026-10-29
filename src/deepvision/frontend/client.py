"""HTTP client for the DeepVision inference API, used by the Streamlit frontend.

The frontend never loads the model: every prediction is a request to the FastAPI service.
This module has no Streamlit import, so it is tested on its own.

- Every response is validated before use (labels, probabilities, batch order); anything
  unexpected becomes ApiError(kind="invalid_response") instead of being displayed.
- Failures are ApiError with a kind (connection, timeout, http, invalid_response) and a
  short, user-facing message. The API's own error messages are path-free by design.
- /predict/batch rejects the whole request when one file is bad ("file 3: ..."). The
  batch helper then drops that file and resends the rest, so one bad image does not sink
  the others. Each retry removes a file, so a chunk of n files needs at most n requests.
- Nothing here logs image bytes or file names; logs carry counts, status codes, and timings.
"""

import logging
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("deepvision.frontend")

CLASS_NAMES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
DEFAULT_API_URL = "http://127.0.0.1:8000"
PER_FILE_ERROR = re.compile(r"^file (\d+)\b")  # the API prefixes per-file errors in batches
PER_FILE_CODES = frozenset({"invalid_image", "unsupported_format", "image_too_large",
                            "payload_too_large"})  # fmt: skip
PROBABILITY_TOLERANCE = 1e-3


class SettingsError(ValueError):
    """Raised when a DEEPVISION_* frontend environment variable is invalid."""


@dataclass(frozen=True)
class FrontendSettings:
    api_url: str = DEFAULT_API_URL
    max_upload_mb: float = 10.0  # keep in line with the API's DEEPVISION_MAX_UPLOAD_MB
    max_batch: int = 16  # files per /predict/batch request (API's DEEPVISION_MAX_BATCH)
    max_files: int = 64  # files per batch upload in the UI (sent in chunks of max_batch)
    max_pixels: int = 40_000_000  # same default as the API
    connect_timeout: float = 3.0
    read_timeout: float = 30.0
    batch_read_timeout: float = 60.0

    def __post_init__(self) -> None:
        validate_api_url(self.api_url)
        for name in ("max_upload_mb", "max_batch", "max_files", "max_pixels", "connect_timeout",
                     "read_timeout", "batch_read_timeout"):  # fmt: skip
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise SettingsError(f"{name} must be a positive number")

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "FrontendSettings":
        environ = os.environ if environ is None else environ
        variables = {
            "api_url": ("DEEPVISION_API_URL", str),
            "max_upload_mb": ("DEEPVISION_MAX_UPLOAD_MB", float),
            "max_batch": ("DEEPVISION_MAX_BATCH", int),
            "max_files": ("DEEPVISION_FRONTEND_MAX_FILES", int),
        }
        values = {}
        for name, (variable, convert) in variables.items():
            raw = environ.get(variable, "").strip()
            if raw:
                try:
                    values[name] = convert(raw)
                except ValueError as exc:
                    raise SettingsError(f"{variable} is not a valid number") from exc
        return cls(**values)


def validate_api_url(url: str) -> httpx.URL:
    """Accept only plain http(s) base URLs without credentials, query, or fragment."""
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, TypeError) as exc:
        raise SettingsError("DEEPVISION_API_URL is not a valid URL") from exc
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise SettingsError("DEEPVISION_API_URL must be an http:// or https:// URL with a host")
    if parsed.userinfo or parsed.query or parsed.fragment:
        raise SettingsError(
            "DEEPVISION_API_URL must not contain credentials, a query, or a fragment"
        )
    return parsed


def display_url(url: str) -> str:
    """scheme://host:port only, for showing in the UI."""
    parsed = httpx.URL(url)
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.host}{port}"


# --- Results ------------------------------------------------------------------------------


class ApiError(Exception):
    """A failed API call. kind: connection, timeout, http, or invalid_response."""

    def __init__(
        self, kind: str, message: str, *, status: int | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.kind, self.message, self.status, self.code = kind, message, status, code


@dataclass(frozen=True)
class Upload:
    name: str
    data: bytes
    content_type: str


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top_k: tuple[tuple[str, float], ...]
    probabilities: dict[str, float]


@dataclass(frozen=True)
class PredictionResult(Prediction):
    model_version: str = ""
    inference_ms: float = 0.0


@dataclass(frozen=True)
class BatchItem:
    index: int  # position in the list given to predict_batch
    prediction: Prediction | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.prediction is not None


@dataclass(frozen=True)
class BatchResult:
    items: tuple[BatchItem, ...]
    model_version: str | None
    requests: int  # API requests made, including retries without rejected files


@dataclass(frozen=True)
class ServiceStatus:
    online: bool  # the API answered /health
    ready: bool  # the model is loaded (/ready returned 200)
    model_version: str | None = None
    message: str = ""


@dataclass(frozen=True)
class ModelInfo:
    model_version: str
    backbone: str
    class_names: tuple[str, ...]
    image_size: int
    epoch: int | None
    training_commit: str
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] | None = None  # held-out clean test set, if available


# --- Response validation ----------------------------------------------------------------


def _invalid(detail: str) -> ApiError:
    logger.warning("invalid API response: %s", detail)
    return ApiError("invalid_response", "The API returned an unexpected response.")


def _number(value, name: str, *, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _invalid(f"{name} is not a number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise _invalid(f"{name} is out of range")
    return float(value)


def parse_prediction(body, classes: Sequence[str] = CLASS_NAMES) -> Prediction:
    """Validate one prediction object of the API's response schema."""
    if not isinstance(body, dict):
        raise _invalid("prediction is not an object")
    probabilities = body.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(classes):
        raise _invalid("probabilities do not list exactly the expected classes")
    probabilities = {
        name: _number(probabilities[name], "probability", low=0, high=1) for name in classes
    }
    if abs(sum(probabilities.values()) - 1) > 1e-2:
        raise _invalid("probabilities do not sum to 1")
    label = body.get("label")
    if label not in probabilities:
        raise _invalid("label is not one of the classes")
    confidence = _number(body.get("confidence"), "confidence", low=0, high=1)
    if (
        abs(confidence - probabilities[label]) > PROBABILITY_TOLERANCE
        or confidence < max(probabilities.values()) - PROBABILITY_TOLERANCE
    ):
        raise _invalid("label and confidence do not match the probabilities")
    top_k = body.get("top_k")
    if not isinstance(top_k, list) or not top_k:
        raise _invalid("top_k is missing")
    ranked = []
    for entry in top_k:
        if not isinstance(entry, dict) or entry.get("label") not in probabilities:
            raise _invalid("top_k entry is malformed")
        ranked.append((entry["label"], _number(entry.get("probability"), "top_k", low=0, high=1)))
    return Prediction(label, confidence, tuple(ranked), probabilities)


def _model_version(body) -> str:
    version = body.get("model_version") if isinstance(body, dict) else None
    if not isinstance(version, str) or not version:
        raise _invalid("model_version is missing")
    return version


# --- Client -------------------------------------------------------------------------------


class ApiClient:
    """Synchronous client; pass transport=httpx.MockTransport(...) in tests."""

    def __init__(
        self,
        settings: FrontendSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        classes: Sequence[str] = CLASS_NAMES,
    ) -> None:
        self.settings = settings
        self.classes = tuple(classes)
        self._http = httpx.Client(
            base_url=settings.api_url,
            transport=transport,
            timeout=httpx.Timeout(settings.read_timeout, connect=settings.connect_timeout),
            follow_redirects=False,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._http.close()

    # -- requests ------------------------------------------------------------------------

    def _request(self, method: str, path: str, *, timeout: float | None = None, **kwargs):
        started = time.perf_counter()
        try:
            response = self._http.request(
                method,
                path,
                timeout=(
                    httpx.Timeout(timeout, connect=self.settings.connect_timeout)
                    if timeout
                    else httpx.USE_CLIENT_DEFAULT
                ),
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            logger.warning("%s %s timed out", method, path)
            raise ApiError(
                "timeout", "The inference API did not respond in time. Try again shortly."
            ) from exc
        except httpx.TransportError as exc:
            logger.warning("%s %s failed: %s", method, path, type(exc).__name__)
            raise ApiError(
                "connection",
                f"Cannot reach the inference API at {display_url(self.settings.api_url)}.",
            ) from exc
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("%s %s -> %d in %.0f ms", method, path, response.status_code, elapsed)
        return response

    def _json(self, response: httpx.Response):
        try:
            return response.json()
        except ValueError:
            raise _invalid("body is not JSON") from None

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        status, code, detail = response.status_code, None, None
        try:
            error = response.json().get("error", {})
            code, detail = error.get("code"), error.get("message")
        except (ValueError, AttributeError):
            pass
        if not isinstance(detail, str):
            detail = None
        if status == 503:
            message = detail or "The model is not loaded yet."
        elif 400 <= status < 500 and detail:
            message = detail
        else:
            message = f"The inference API failed to process the request (HTTP {status})."
        raise ApiError(
            "http", message[:300], status=status, code=code if isinstance(code, str) else None
        )

    # -- endpoints -----------------------------------------------------------------------

    def status(self) -> ServiceStatus:
        """Liveness (/health) and readiness (/ready); never raises."""
        try:
            health = self._request("GET", "/health")
            self._raise_for_status(health)
        except ApiError as exc:
            return ServiceStatus(online=False, ready=False, message=exc.message)
        try:
            ready = self._request("GET", "/ready")
            self._raise_for_status(ready)
            return ServiceStatus(True, True, _model_version(self._json(ready)), "Model ready")
        except ApiError as exc:
            return ServiceStatus(online=True, ready=False, message=exc.message)

    def model_info(self) -> ModelInfo:
        response = self._request("GET", "/model")
        self._raise_for_status(response)
        body = self._json(response)
        if not isinstance(body, dict):
            raise _invalid("model info is not an object")
        try:
            classes = tuple(body["class_names"])
            validation = {
                k: float(v)
                for k, v in body.get("validation_metrics", {}).items()
                if isinstance(v, int | float)
            }
            held_out = (body.get("evaluation_metrics") or {}).get("test")
            test = (
                {k: float(v) for k, v in held_out.items() if isinstance(v, int | float)}
                if isinstance(held_out, dict)
                else None
            )
            return ModelInfo(
                model_version=_model_version(body),
                backbone=str(body["backbone"]),
                class_names=classes,
                image_size=int(body["image_size"]),
                epoch=body.get("epoch") if isinstance(body.get("epoch"), int) else None,
                training_commit=str(body.get("training_commit", "unknown")),
                validation_metrics=validation,
                test_metrics=test,
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            raise _invalid("model info does not match the schema") from None

    def predict(self, upload: Upload) -> PredictionResult:
        response = self._request(
            "POST", "/predict", files={"file": (upload.name, upload.data, upload.content_type)}
        )
        self._raise_for_status(response)
        body = self._json(response)
        prediction = parse_prediction(body, self.classes)
        inference_ms = _number(body.get("inference_ms"), "inference_ms", low=0)
        return PredictionResult(
            prediction.label,
            prediction.confidence,
            prediction.top_k,
            prediction.probabilities,
            model_version=_model_version(body),
            inference_ms=inference_ms,
        )

    def _post_batch(self, uploads: Sequence[Upload]) -> tuple[list[Prediction], str]:
        files = [("files", (u.name, u.data, u.content_type)) for u in uploads]
        response = self._request(
            "POST", "/predict/batch", files=files, timeout=self.settings.batch_read_timeout
        )
        self._raise_for_status(response)
        body = self._json(response)
        items = body.get("predictions") if isinstance(body, dict) else None
        if not isinstance(items, list) or len(items) != len(uploads):
            raise _invalid("batch does not return one prediction per file")
        if [item.get("index") if isinstance(item, dict) else None for item in items] != list(
            range(len(uploads))
        ):
            raise _invalid("batch predictions are not in request order")
        return [parse_prediction(item, self.classes) for item in items], _model_version(body)

    def predict_batch(self, uploads: Sequence[Upload]) -> BatchResult:
        """Predict every upload, isolating files the API rejects.

        Uploads are sent in chunks of max_batch. If the API rejects a chunk because of one
        file, that file is marked failed and the rest of the chunk is resent. Every retry
        removes one file, so a chunk of n files takes at most n requests (the whole batch
        at most len(uploads)). The `range` cap below enforces the same bound a second time
        in case that invariant is ever broken.
        """
        results: dict[int, BatchItem] = {}
        version, requests = None, 0
        size = self.settings.max_batch
        for start in range(0, len(uploads), size):
            remaining = list(range(start, min(start + size, len(uploads))))
            for _attempt in range(len(remaining)):  # hard cap: one request per file
                requests += 1
                try:
                    predictions, version = self._post_batch([uploads[i] for i in remaining])
                except ApiError as exc:
                    match = PER_FILE_ERROR.match(exc.message) if exc.kind == "http" else None
                    position = int(match.group(1)) if match else None
                    if (
                        exc.code in PER_FILE_CODES
                        and position is not None
                        and position < len(remaining)
                    ):
                        failed = remaining.pop(position)
                        reason = exc.message[match.end() :].lstrip(": ") or exc.message
                        results[failed] = BatchItem(failed, error=_capitalize(reason))
                        if remaining:
                            continue
                        break
                    for i in remaining:  # not attributable to one file: the chunk fails
                        results[i] = BatchItem(i, error=exc.message)
                    break
                for i, prediction in zip(remaining, predictions, strict=True):
                    results[i] = BatchItem(i, prediction=prediction)
                break
            for i in range(start, min(start + size, len(uploads))):
                results.setdefault(i, BatchItem(i, error="The image could not be processed."))
        failed = sum(not item.ok for item in results.values())
        logger.info("batch of %d files: %d failed, %d requests", len(uploads), failed, requests)
        return BatchResult(tuple(results[i] for i in range(len(uploads))), version, requests)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]
