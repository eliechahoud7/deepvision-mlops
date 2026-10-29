import json
import logging

import httpx
import pytest

from deepvision.frontend.client import (
    CLASS_NAMES,
    ApiClient,
    ApiError,
    FrontendSettings,
    SettingsError,
    Upload,
    display_url,
    parse_prediction,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels-not-sent-to-logs"


def probabilities(label="forest", confidence=0.9) -> dict:
    rest = (1 - confidence) / (len(CLASS_NAMES) - 1)
    return {name: (confidence if name == label else rest) for name in CLASS_NAMES}


def prediction_body(label="forest", confidence=0.9, **extra) -> dict:
    probs = probabilities(label, confidence)
    top = sorted(probs.items(), key=lambda item: -item[1])[:3]
    return {
        "label": label,
        "confidence": confidence,
        "top_k": [{"label": k, "probability": v} for k, v in top],
        "probabilities": probs,
        **extra,
    }


def error(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


def make_client(handler, **settings) -> ApiClient:
    return ApiClient(
        FrontendSettings(api_url="http://api.test:8000", **settings),
        transport=httpx.MockTransport(handler),
    )


def uploads(n: int) -> list[Upload]:
    return [Upload(f"img{i}.png", PNG + bytes([i]), "image/png") for i in range(n)]


def sent_names(request: httpx.Request) -> list[str]:
    return [part.split(b'filename="')[1].split(b'"')[0].decode()
            for part in request.content.split(b"\r\n--") if b'filename="' in part]  # fmt: skip


# --- Settings -----------------------------------------------------------------------------


def test_settings_defaults_match_the_api():
    settings = FrontendSettings.from_env({})
    assert settings.api_url == "http://127.0.0.1:8000"
    assert (settings.max_upload_mb, settings.max_batch, settings.max_pixels) == (10, 16, 40_000_000)


def test_settings_from_environment():
    settings = FrontendSettings.from_env({
        "DEEPVISION_API_URL": "http://api:8000", "DEEPVISION_MAX_UPLOAD_MB": "5",
        "DEEPVISION_MAX_BATCH": "8", "DEEPVISION_FRONTEND_MAX_FILES": "20",
    })  # fmt: skip
    assert (settings.api_url, settings.max_upload_bytes) == ("http://api:8000", 5 * 1024 * 1024)
    assert (settings.max_batch, settings.max_files) == (8, 20)


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"DEEPVISION_API_URL": "ftp://api:8000"}, "http:// or https://"),
        ({"DEEPVISION_API_URL": "http://user:secret@api:8000"}, "credentials"),
        ({"DEEPVISION_API_URL": "http://api:8000/?x=1"}, "query"),
        ({"DEEPVISION_API_URL": "not a url"}, "http"),
        ({"DEEPVISION_MAX_BATCH": "many"}, "not a valid number"),
        ({"DEEPVISION_MAX_BATCH": "0"}, "positive"),
        ({"DEEPVISION_MAX_UPLOAD_MB": "nan"}, "positive"),
    ],
)
def test_invalid_settings_are_rejected(environ, message):
    with pytest.raises(SettingsError, match=message):
        FrontendSettings.from_env(environ)


def test_display_url_hides_paths():
    assert display_url("http://api:8000") == "http://api:8000"
    assert display_url("https://example.org/base/") == "https://example.org"


# --- Status and model information ---------------------------------------------------------


def test_status_ready():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"status": "ready", "model_version": "resnet18-abc"})

    status = make_client(handler).status()
    assert (status.online, status.ready, status.model_version) == (True, True, "resnet18-abc")


def test_status_online_but_model_not_loaded():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return error(503, "model_unavailable", "the model is not ready: checkpoint missing")

    status = make_client(handler).status()
    assert (status.online, status.ready) == (True, False)
    assert "checkpoint missing" in status.message


def test_status_offline_never_raises():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    status = make_client(handler).status()
    assert (status.online, status.ready) == (False, False)
    assert "Cannot reach the inference API at http://api.test:8000" in status.message


def test_model_info():
    body = {
        "model_version": "resnet18-abc", "backbone": "resnet18", "class_names": list(CLASS_NAMES),
        "num_classes": 6, "image_size": 224, "epoch": 8, "checkpoint_md5": "x" * 32,
        "training_commit": "bff616d", "training_tree_dirty": "false", "mlflow_run_id": None,
        "device": "cpu", "validation_metrics": {"macro_f1": 0.949, "loss": 0.2},
        "evaluation_metrics": {"test": {"images": 2997, "macro_f1": 0.94, "accuracy": 0.939}},
    }  # fmt: skip
    info = make_client(lambda r: httpx.Response(200, json=body)).model_info()
    assert info.backbone == "resnet18" and info.class_names == CLASS_NAMES
    assert info.test_metrics["macro_f1"] == 0.94 and info.validation_metrics["macro_f1"] == 0.949


def test_model_info_without_held_out_metrics_and_malformed():
    body = {"model_version": "v", "backbone": "b", "class_names": ["a"], "image_size": 32,
            "evaluation_metrics": None}  # fmt: skip
    assert make_client(lambda r: httpx.Response(200, json=body)).model_info().test_metrics is None
    with pytest.raises(ApiError) as exc_info:
        make_client(lambda r: httpx.Response(200, json={"backbone": "b"})).model_info()
    assert exc_info.value.kind == "invalid_response"


# --- Single prediction ----------------------------------------------------------------------


def test_predict_sends_the_file_and_parses_the_response():
    seen = {}

    def handler(request):
        seen["path"], seen["names"] = request.url.path, sent_names(request)
        return httpx.Response(200, json=prediction_body(model_version="v1", inference_ms=31.5))

    result = make_client(handler).predict(Upload("photo.png", PNG, "image/png"))
    assert seen == {"path": "/predict", "names": ["photo.png"]}
    assert (result.label, result.confidence, result.model_version) == ("forest", 0.9, "v1")
    assert result.inference_ms == 31.5 and set(result.probabilities) == set(CLASS_NAMES)
    assert result.top_k[0] == ("forest", 0.9)


@pytest.mark.parametrize(
    ("status", "code", "api_message", "expected"),
    [
        (400, "invalid_image", "the image: the image data is corrupt", "the image data is corrupt"),
        (413, "payload_too_large", "the request body exceeds the limit", "exceeds the limit"),
        (415, "unsupported_format", "the image: unsupported image format GIF", "GIF"),
        (422, "invalid_request", "the request is invalid", "the request is invalid"),
        (503, "model_unavailable", "the model is not ready: x", "the model is not ready"),
        (
            500,
            "internal_error",
            "internal server error",
            "failed to process the request (HTTP 500)",
        ),
    ],
)
def test_predict_http_errors(status, code, api_message, expected):
    client = make_client(lambda r: error(status, code, api_message))
    with pytest.raises(ApiError) as exc_info:
        client.predict(Upload("a.png", PNG, "image/png"))
    assert exc_info.value.kind == "http" and exc_info.value.status == status
    assert expected in exc_info.value.message and exc_info.value.code == code


def test_non_json_error_body_and_proxy_errors():
    client = make_client(lambda r: httpx.Response(502, text="<html>Bad gateway</html>"))
    with pytest.raises(ApiError, match="HTTP 502") as exc_info:
        client.predict(Upload("a.png", PNG, "image/png"))
    assert "<html>" not in exc_info.value.message


def test_timeout_and_connection_errors():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("refused", request=request)

    for handler, kind, text in ((slow, "timeout", "did not respond in time"),
                                (refused, "connection", "Cannot reach")):  # fmt: skip
        with pytest.raises(ApiError, match=text) as exc_info:
            make_client(handler).predict(Upload("a.png", PNG, "image/png"))
        assert exc_info.value.kind == kind


def test_redirects_are_not_followed():
    client = make_client(lambda r: httpx.Response(307, headers={"location": "http://evil.test/"}))
    with pytest.raises(ApiError, match="HTTP 307"):
        client.predict(Upload("a.png", PNG, "image/png"))


VALID = prediction_body(model_version="v1", inference_ms=12.0)


def without(key: str) -> dict:
    return {k: v for k, v in VALID.items() if k != key}


def json_response(body) -> httpx.Response:
    # Raw JSON text: Python's decoder accepts NaN, which the client must still reject.
    content = json.dumps(body, allow_nan=True).encode()
    return httpx.Response(200, content=content, headers={"content-type": "application/json"})


def test_the_valid_base_response_is_accepted():
    # Control for the malformed cases below: each one breaks exactly one thing in VALID.
    assert make_client(lambda r: json_response(VALID)).predict(Upload("a.png", PNG, "image/png"))


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("not json", "body is not JSON"),
        ([], "prediction is not an object"),
        (VALID | {"label": "desert"}, "label is not one of the classes"),
        (VALID | {"confidence": 1.5}, "confidence is out of range"),
        (VALID | {"confidence": "high"}, "confidence is not a number"),
        (VALID | {"confidence": True}, "confidence is not a number"),
        (VALID | {"probabilities": {"forest": 1.0}}, "exactly the expected classes"),
        (
            VALID | {"probabilities": probabilities() | {"desert": 0.0}},
            "exactly the expected classes",
        ),
        (
            VALID | {"probabilities": probabilities() | {"sea": float("nan")}},
            "probability is not a number",
        ),
        (VALID | {"probabilities": probabilities() | {"sea": -0.1}}, "probability is out of range"),
        # Every class at 0.9: label, confidence, and top_k are consistent; only the sum is wrong.
        (VALID | {"probabilities": {n: 0.9 for n in CLASS_NAMES}}, "do not sum to 1"),
        (VALID | {"confidence": 0.02}, "do not match the probabilities"),
        (
            prediction_body("sea", 0.9, model_version="v", inference_ms=1)
            | {"label": "forest", "confidence": probabilities("sea", 0.9)["forest"]},
            "do not match the probabilities",
        ),  # fmt: skip
        (VALID | {"top_k": []}, "top_k is missing"),
        (VALID | {"top_k": [{"label": "desert", "probability": 0.1}]}, "top_k entry is malformed"),
        (VALID | {"top_k": [{"label": "forest", "probability": 7}]}, "top_k is out of range"),
        (without("model_version"), "model_version is missing"),
        (VALID | {"model_version": ""}, "model_version is missing"),
        (without("inference_ms"), "inference_ms is not a number"),
        (VALID | {"inference_ms": -1}, "inference_ms is out of range"),
    ],
)
def test_malformed_responses_are_rejected(body, reason, caplog):
    def handler(request):
        return httpx.Response(200, text="oops") if body == "not json" else json_response(body)

    with caplog.at_level(logging.WARNING), pytest.raises(ApiError) as exc_info:
        make_client(handler).predict(Upload("a.png", PNG, "image/png"))
    assert exc_info.value.kind == "invalid_response"
    assert exc_info.value.message == "The API returned an unexpected response."
    assert reason in caplog.text  # the intended check fired, not an unrelated one


def test_parse_prediction_accepts_the_api_schema():
    prediction = parse_prediction(prediction_body("glacier", 0.7))
    assert prediction.label == "glacier" and abs(sum(prediction.probabilities.values()) - 1) < 1e-9


# --- Batch ----------------------------------------------------------------------------------


def batch_response(n: int, labels=None, **overrides) -> httpx.Response:
    labels = labels or ["forest"] * n
    items = [prediction_body(label) | {"index": i} for i, label in enumerate(labels)]
    body = {"predictions": items, "model_version": "v1", "inference_ms": 10.0} | overrides
    return httpx.Response(200, json=body)


class FakeBatchApi:
    """Rejects any request containing a file whose name is in `bad`, like the real API:
    it reports the first bad file's position in the request."""

    def __init__(self, bad=(), code="invalid_image", max_batch=16):
        self.bad, self.code, self.max_batch, self.requests = set(bad), code, max_batch, []

    def __call__(self, request):
        names = sent_names(request)
        self.requests.append(names)
        if len(names) > self.max_batch:
            return error(413, "batch_too_large", f"{len(names)} files sent; the limit is 16")
        for position, name in enumerate(names):
            if name in self.bad:
                return error(400, self.code, f"file {position}: the image data is corrupt")
        return batch_response(len(names), labels=["sea" if "3" in n else "forest" for n in names])


def test_batch_success_in_one_request():
    api = FakeBatchApi()
    result = make_client(api).predict_batch(uploads(5))
    assert result.requests == 1 and len(api.requests) == 1
    assert [item.prediction.label for item in result.items] == ["forest"] * 3 + ["sea", "forest"]
    assert result.model_version == "v1" and all(item.ok for item in result.items)


def test_batch_isolates_bad_files_and_keeps_order():
    api = FakeBatchApi(bad={"img1.png", "img3.png"})
    result = make_client(api).predict_batch(uploads(5))
    assert [item.ok for item in result.items] == [True, False, True, False, True]
    assert result.items[1].error == "The image data is corrupt"
    assert [item.index for item in result.items] == list(range(5))
    # The good files were predicted from the final request, which excluded both bad ones.
    assert api.requests[-1] == ["img0.png", "img2.png", "img4.png"]
    assert result.requests == 3


def test_batch_retries_are_bounded_when_every_file_is_bad():
    names = [u.name for u in uploads(6)]
    api = FakeBatchApi(bad=set(names))
    result = make_client(api).predict_batch(uploads(6))
    assert not any(item.ok for item in result.items)
    assert result.requests == len(api.requests) == 6  # one request per file, never more


def test_batch_retries_stop_on_an_error_that_names_no_file():
    calls = []

    def handler(request):
        calls.append(1)
        return error(503, "model_unavailable", "the model is not ready: checkpoint missing")

    result = make_client(handler).predict_batch(uploads(4))
    assert len(calls) == 1 and result.requests == 1
    assert all("not ready" in item.error for item in result.items)


def test_batch_ignores_an_out_of_range_file_position():
    calls = []

    def handler(request):
        calls.append(1)
        return error(400, "invalid_image", "file 9: corrupt")  # only 2 files were sent

    result = make_client(handler).predict_batch(uploads(2))
    assert len(calls) == 1 and not any(item.ok for item in result.items)


def test_batch_is_chunked_by_max_batch():
    api = FakeBatchApi(max_batch=4)
    result = make_client(api, max_batch=4).predict_batch(uploads(10))
    assert [len(names) for names in api.requests] == [4, 4, 2]
    assert all(item.ok for item in result.items) and len(result.items) == 10


def test_batch_rejects_mismatched_responses():
    responses = iter([batch_response(1), batch_response(3, predictions=[])])
    client = make_client(lambda r: next(responses))
    result = client.predict_batch(uploads(3))  # API answered with 1 prediction for 3 files
    assert not any(item.ok for item in result.items)
    assert result.items[0].error == "The API returned an unexpected response."


def test_batch_rejects_out_of_order_indices():
    body = json.loads(batch_response(2).content)
    body["predictions"].reverse()
    result = make_client(lambda r: httpx.Response(200, json=body)).predict_batch(uploads(2))
    assert not any(item.ok for item in result.items)


def test_logs_never_contain_image_bytes_or_file_names(caplog):
    batch_api = FakeBatchApi(bad={"img1.png"})

    def handler(request):
        if request.url.path == "/predict":
            return httpx.Response(200, json=prediction_body(model_version="v", inference_ms=1))
        return batch_api(request)

    with caplog.at_level(logging.DEBUG):
        client = make_client(handler)
        client.predict_batch(uploads(3))
        client.predict(Upload("secret-name.png", PNG, "image/png"))
    text = caplog.text
    assert "batch of 3 files: 1 failed, 2 requests" in text
    assert "img1.png" not in text and "secret-name" not in text and "pixels-not-sent" not in text
