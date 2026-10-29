import json
import logging
import re

import pytest
from conftest import CLASSES, image_bytes
from fastapi.testclient import TestClient
from starlette.formparsers import MultiPartParser

from deepvision.api.app import create_app
from deepvision.api.settings import Settings, SettingsError
from deepvision.inference.predictor import file_md5

PNG = image_bytes("PNG", size=(40, 30))


def make_client(tmp_path, checkpoint=None, **overrides) -> TestClient:
    settings = Settings(
        checkpoint=checkpoint if checkpoint is not None else tmp_path / "missing.pt",
        evaluation_metrics=tmp_path / "metrics.json",
        **({"max_batch": 4} | overrides),
    )
    return TestClient(create_app(settings), raise_server_exceptions=False)


@pytest.fixture
def client(tmp_path, checkpoint):
    with make_client(tmp_path, checkpoint) as client:
        yield client


@pytest.fixture
def unready(tmp_path):
    with make_client(tmp_path) as client:
        yield client


def upload(data: bytes = PNG, name: str = "image.png", field: str = "file"):
    return {field: (name, data, "application/octet-stream")}


def assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"} and body["error"]["code"] == code
    assert "Traceback" not in response.text
    return body["error"]


# --- Service endpoints ----------------------------------------------------------------


def test_health_and_readiness(client):
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    assert ready.json()["model_version"].startswith("mobilenet_v3_small-")


def test_readiness_failure_is_distinct_from_liveness(unready, tmp_path):
    assert unready.get("/health").status_code == 200
    for response in (
        unready.get("/ready"),
        unready.get("/model"),
        unready.post("/predict", files=upload()),
    ):
        error = assert_error(response, 503, "model_unavailable")
        assert "checkpoint is not available" in error["message"]
        assert str(tmp_path) not in response.text and "missing.pt" not in response.text
    assert "deepvision_model_ready 0.0" in unready.get("/metrics").text


def test_unreadable_checkpoint_is_reported_without_paths(tmp_path, caplog):
    bad = tmp_path / "secret-location" / "best.pt"
    bad.parent.mkdir()
    bad.write_bytes(b"corrupt")
    with caplog.at_level(logging.INFO, logger="deepvision.api"), make_client(tmp_path, bad) as c:
        error = assert_error(c.get("/ready"), 503, "model_unavailable")
    assert "could not be read" in error["message"]
    assert "secret-location" not in caplog.text and "model not ready: unreadable" in caplog.text


def test_unavailable_device_and_unexpected_load_errors(tmp_path, checkpoint, monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with make_client(tmp_path, checkpoint, device="cuda") as c:
        error = assert_error(c.get("/ready"), 503, "model_unavailable")
    assert "inference device is not available" in error["message"]

    def failing_factory(*args, **kwargs):
        raise ValueError(f"unexpected at {tmp_path}")

    app = create_app(Settings(checkpoint=checkpoint), predictor_factory=failing_factory)
    with caplog.at_level(logging.INFO, logger="deepvision.api"), TestClient(app) as c:
        error = assert_error(c.get("/ready"), 503, "model_unavailable")
        assert c.get("/health").status_code == 200
    assert "could not be loaded" in error["message"]
    assert "model load failed with ValueError" in caplog.text and str(tmp_path) not in caplog.text


def test_model_metadata(tmp_path, checkpoint):
    report = tmp_path / "metrics.json"
    report.write_text(
        json.dumps({"checkpoint": {"md5": file_md5(checkpoint)}, "test": {"macro_f1": 0.94}})
    )
    with make_client(tmp_path, checkpoint) as c:
        info = c.get("/model").json()

    assert info["class_names"] == CLASSES and info["num_classes"] == 6
    assert (info["backbone"], info["image_size"], info["epoch"]) == ("mobilenet_v3_small", 32, 3)
    assert info["training_commit"] == "a" * 40 and info["device"] == "cpu"
    assert info["validation_metrics"]["macro_f1"] == 0.8
    assert info["evaluation_metrics"]["test"]["macro_f1"] == 0.94
    assert str(tmp_path) not in json.dumps(info) and "best.pt" not in json.dumps(info)


def test_openapi_documents_every_endpoint(client):
    schema = client.get("/openapi.json").json()
    assert {"/health", "/ready", "/model", "/predict", "/predict/batch", "/metrics"} <= set(
        schema["paths"]
    )
    predict = schema["paths"]["/predict"]["post"]["responses"]
    assert {"200", "400", "413", "415", "422", "503"} <= set(predict)
    assert client.get("/docs").status_code == 200


# --- Prediction ---------------------------------------------------------------------


def check_result(result: dict) -> None:
    assert result["label"] in CLASSES
    assert list(result["probabilities"]) == CLASSES
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    scores = [item["probability"] for item in result["top_k"]]
    assert len(scores) == 3 and scores == sorted(scores, reverse=True)
    assert result["top_k"][0]["label"] == result["label"]
    assert result["confidence"] == pytest.approx(max(result["probabilities"].values()), abs=1e-6)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "BMP"])
def test_predict_accepts_supported_formats(client, fmt):
    response = client.post("/predict", files=upload(image_bytes(fmt), name="x.bin"))

    assert response.status_code == 200, response.text
    body = response.json()
    check_result(body)
    assert body["model_version"].startswith("mobilenet_v3_small-") and body["inference_ms"] > 0
    assert set(body) == {
        "label", "confidence", "top_k", "probabilities", "model_version", "inference_ms",
    }  # fmt: skip


def test_batch_preserves_request_order(client):
    images = [image_bytes(seed=i, size=(30 + 5 * i, 30)) for i in range(4)]
    files = [("files", (f"{i}.png", data, "image/png")) for i, data in enumerate(images)]

    batch = client.post("/predict/batch", files=files)

    assert batch.status_code == 200, batch.text
    results = batch.json()["predictions"]
    assert [r["index"] for r in results] == [0, 1, 2, 3]
    for result, data in zip(results, images, strict=True):
        check_result(result)
        single = client.post("/predict", files=upload(data)).json()
        assert result["label"] == single["label"]
        for name in CLASSES:
            assert result["probabilities"][name] == pytest.approx(
                single["probabilities"][name], abs=1e-5
            )


# --- Errors -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (image_bytes("GIF", size=(32, 32)), 415, "unsupported_format"),
        (b"not an image", 400, "invalid_image"),
        (b"", 400, "invalid_image"),
        (PNG[:100], 400, "invalid_image"),
    ],
    ids=["gif", "garbage", "empty", "truncated"],
)
def test_bad_images_get_structured_errors(client, data, status, code):
    assert_error(client.post("/predict", files=upload(data)), status, code)


def test_missing_file_and_unknown_routes(client):
    error = assert_error(client.post("/predict"), 422, "invalid_request")
    assert error["details"][0]["loc"] == ["body", "file"]
    assert_error(client.get("/nope"), 404, "not_found")
    assert_error(client.get("/predict"), 405, "method_not_allowed")


def test_upload_size_limits(tmp_path, checkpoint):
    big = image_bytes("PNG", size=(120, 120))  # ~43 KB of noise
    with make_client(tmp_path, checkpoint, max_upload_mb=0.01) as c:  # 10 KiB per file
        # Within the batch body limit, but the file itself is too large.
        error = assert_error(
            c.post("/predict/batch", files=[("files", ("a.png", big, "image/png"))]),
            413,
            "payload_too_large",
        )
        assert "file 0 exceeds" in error["message"]
        # The whole single-image body is over the limit (Content-Length check).
        three = [("files", ("a.png", big, "image/png"))] * 3
        assert_error(c.post("/predict", files=upload(big + big)), 413, "payload_too_large")
        assert_error(c.post("/predict/batch", files=three), 413, "payload_too_large")


def test_streamed_body_over_the_limit_is_cut_off(tmp_path, checkpoint):
    data = image_bytes("PNG", size=(120, 120))
    body = (
        b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n'
        b"Content-Type: image/png\r\n\r\n" + data * 2 + b"\r\n--x--\r\n"
    )

    def chunks():  # no Content-Length: the size is only known while receiving
        for start in range(0, len(body), 8192):
            yield body[start : start + 8192]

    with make_client(tmp_path, checkpoint, max_upload_mb=0.01) as c:
        response = c.post(
            "/predict",
            content=chunks(),
            headers={"content-type": "multipart/form-data; boundary=x"},
        )
    assert_error(response, 413, "payload_too_large")


def test_batch_and_pixel_limits(tmp_path, checkpoint):
    with make_client(tmp_path, checkpoint, max_batch=2, max_pixels=1000) as c:
        three = [("files", (f"{i}.png", PNG, "image/png")) for i in range(3)]
        error = assert_error(c.post("/predict/batch", files=three), 413, "batch_too_large")
        assert "the batch limit is 2" in error["message"]
        error = assert_error(c.post("/predict", files=upload(PNG)), 413, "image_too_large")
        assert "1200 pixels" in error["message"]


def test_internal_errors_hide_details(client, tmp_path, caplog):
    def broken(images):
        raise RuntimeError(f"secret detail at {tmp_path}")

    client.app.state.predictor.predict = broken
    with caplog.at_level(logging.INFO, logger="deepvision.api"):
        response = client.post("/predict", files=upload())

    error = assert_error(response, 500, "internal_error")
    assert error["message"] == "inference failed"
    assert "secret" not in response.text and "secret" not in caplog.text
    assert "inference failed with RuntimeError" in caplog.text


def test_logs_contain_no_image_bytes_or_paths(client, checkpoint, caplog):
    with caplog.at_level(logging.INFO, logger="deepvision.api"):
        client.post("/predict", files=upload())
    assert "POST /predict -> 200" in caplog.text and "predicted " in caplog.text
    assert str(checkpoint.parent) not in caplog.text and "\\x89PNG" not in caplog.text


# --- Metrics and settings -----------------------------------------------------------


def metric(text: str, name: str, labels: str = "") -> float:
    match = re.search(rf"^{re.escape(name + labels)} (\S+)$", text, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def test_prometheus_metrics(client):
    client.post("/predict", files=upload())
    client.post("/predict", files=upload(image_bytes(seed=1)))
    client.post("/predict", files=upload(b"bad"))
    text = client.get("/metrics").text

    assert metric(text, "deepvision_requests_total", '{endpoint="/predict",status="200"}') == 2
    assert metric(text, "deepvision_requests_total", '{endpoint="/predict",status="400"}') == 1
    assert metric(text, "deepvision_request_latency_seconds_count", '{endpoint="/predict"}') == 3
    assert metric(text, "deepvision_inference_latency_seconds_count") == 2
    assert metric(text, "deepvision_prediction_confidence_count") == 2
    assert (
        sum(metric(text, "deepvision_predictions_total", f'{{label="{name}"}}') for name in CLASSES)
        == 2
    )
    assert metric(text, "deepvision_model_ready") == 1
    assert re.search(
        r'^deepvision_model_info\{.*backbone="mobilenet_v3_small".*\} 1\.0$', text, re.M
    )


def test_each_app_has_its_own_metrics(tmp_path, checkpoint):
    with make_client(tmp_path, checkpoint) as first, make_client(tmp_path, checkpoint) as second:
        first.post("/predict", files=upload())
        text = second.get("/metrics").text
    assert metric(text, "deepvision_requests_total", '{endpoint="/predict",status="200"}') == 0


def test_uploads_stay_in_memory(tmp_path, checkpoint):
    create_app(Settings(checkpoint=checkpoint, max_upload_mb=2, max_batch=4))
    assert MultiPartParser.spool_max_size >= 8 * 1024 * 1024


def test_settings_from_environment():
    settings = Settings.from_env(
        {
            "DEEPVISION_CHECKPOINT": "model.pt",
            "DEEPVISION_DEVICE": "CUDA",
            "DEEPVISION_MAX_UPLOAD_MB": "2.5",
            "DEEPVISION_MAX_BATCH": "8",
            "DEEPVISION_TOP_K": "2",
            "DEEPVISION_NUM_THREADS": "",
        }
    )
    assert (str(settings.checkpoint), settings.device, settings.max_batch) == (
        "model.pt",
        "cuda",
        8,
    )
    assert settings.max_upload_bytes == int(2.5 * 1024 * 1024) and settings.num_threads is None
    assert Settings.from_env({}) == Settings()


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("DEEPVISION_DEVICE", "tpu", "DEVICE"),
        ("DEEPVISION_MAX_BATCH", "zero", "not a valid number"),
        ("DEEPVISION_MAX_BATCH", "0", "MAX_BATCH must be positive"),
        ("DEEPVISION_NUM_THREADS", "0", "NUM_THREADS"),
    ],
)
def test_invalid_settings_are_rejected(variable, value, message):
    with pytest.raises(SettingsError, match=message):
        Settings.from_env({variable: value})


def test_model_load_is_attempted_once_at_startup(tmp_path, checkpoint):
    calls = []

    def factory(*args, **kwargs):
        from deepvision.inference.predictor import Predictor

        calls.append(1)
        return Predictor(*args, **kwargs)

    app = create_app(Settings(checkpoint=checkpoint), predictor_factory=factory)
    with TestClient(app) as c:
        for _ in range(3):
            c.post("/predict", files=upload())
    assert calls == [1]
