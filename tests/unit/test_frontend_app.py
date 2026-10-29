"""UI tests: the real Streamlit app and ApiClient against an in-memory fake API."""

import logging
import tomllib
from pathlib import Path

import httpx
import pytest
import streamlit as st
from conftest import CLASSES, image_bytes
from streamlit.testing.v1 import AppTest

from deepvision.frontend import app
from deepvision.frontend import client as client_module

REPO_ROOT = Path(__file__).resolve().parents[2]
APP = REPO_ROOT / "src" / "deepvision" / "frontend" / "app.py"
PNG = image_bytes("PNG", size=(40, 30))
JPEG = image_bytes("JPEG", size=(40, 30))


def prediction(label="forest", confidence=0.9, index=None) -> dict:
    rest = (1 - confidence) / 5
    probs = {name: (confidence if name == label else rest) for name in CLASSES}
    body = {"label": label, "confidence": confidence, "probabilities": probs,
            "top_k": [{"label": label, "probability": confidence}]}  # fmt: skip
    return body | ({"index": index} if index is not None else {})


def api_error(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


class FakeApi:
    """Minimal stand-in for the FastAPI service (same schemas and error envelope)."""

    def __init__(self, *, online=True, ready=True, bad_names=(), predict_status=200, body=None):
        self.online, self.ready, self.bad_names = online, ready, set(bad_names)
        self.predict_status, self.body = predict_status, body
        self.requests: list[tuple[str, list[str]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if not self.online:
            raise httpx.ConnectError("refused", request=request)
        path = request.url.path
        names = [p.split(b'filename="')[1].split(b'"')[0].decode()
                 for p in request.content.split(b"\r\n--") if b'filename="' in p]  # fmt: skip
        self.requests.append((path, names))
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if not self.ready and path != "/health":
            return api_error(503, "model_unavailable", "the model is not ready: checkpoint missing")
        if path == "/ready":
            return httpx.Response(200, json={"status": "ready", "model_version": "resnet18-test"})
        if path == "/model":
            return httpx.Response(200, json={
                "model_version": "resnet18-test", "backbone": "resnet18", "class_names": CLASSES,
                "image_size": 224, "epoch": 8, "training_commit": "bff616d4356f",
                "validation_metrics": {"macro_f1": 0.949},
                "evaluation_metrics": {
                    "test": {"images": 2997, "macro_f1": 0.940, "accuracy": 0.939}
                },
            })  # fmt: skip
        if path == "/predict":
            if self.predict_status != 200:
                return api_error(self.predict_status, "internal_error", "internal server error")
            return httpx.Response(200, json=self.body or prediction() | {
                "model_version": "resnet18-test", "inference_ms": 31.0})  # fmt: skip
        if path == "/predict/batch":
            for position, name in enumerate(names):
                if name in self.bad_names:
                    message = f"file {position}: the image data is corrupt"
                    return api_error(400, "invalid_image", message)
            items = [
                prediction("sea" if "sea" in n else "forest", 0.8, i) for i, n in enumerate(names)
            ]
            return httpx.Response(200, json={"predictions": items, "model_version": "resnet18-test",
                                             "inference_ms": 50.0})  # fmt: skip
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "Not Found"}})

    def calls(self, path: str) -> list[list[str]]:
        return [names for p, names in self.requests if p == path]


class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def records():
    logger = logging.getLogger("deepvision.frontend")
    handler = Records()
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)


@pytest.fixture
def run_app(monkeypatch):
    """Returns run(fake_api) -> AppTest, with the API client wired to the fake API."""
    real = client_module.ApiClient

    def run(api: FakeApi, **env) -> AppTest:
        monkeypatch.setenv("DEEPVISION_API_URL", env.get("url", "http://api.test:8000"))
        transport = httpx.MockTransport(api)
        monkeypatch.setattr(client_module, "ApiClient", lambda s: real(s, transport=transport))
        st.cache_data.clear()
        st.cache_resource.clear()
        at = AppTest.from_file(str(APP), default_timeout=60)
        return at.run()

    yield run
    st.cache_data.clear()
    st.cache_resource.clear()


def plain(value) -> str:
    """Rendered text of a markdown value: escaping backslashes are not displayed."""
    return str(value).replace("\\", "")


def texts(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "error", "warning", "info", "success", "metric"):
        for element in getattr(at, kind):
            parts.append(
                str(getattr(element, "value", "")) + " " + str(getattr(element, "label", ""))
            )
    return plain("\n".join(parts))


def upload(at: AppTest, key: str, files) -> AppTest:
    at.file_uploader(key=key).set_value(files)
    return at.run()


# --- Service status -------------------------------------------------------------------------


def test_ready_api_shows_status_and_model_card(run_app):
    at = run_app(FakeApi())
    assert not at.exception
    assert any("model ready" in s.value for s in at.sidebar.success)
    text = texts(at)
    assert "resnet18" in text and "0.940" in text
    assert "http://api.test:8000" not in text  # the connection section was removed


def test_model_version_is_shown_without_escape_backslashes(run_app):
    # Raw markdown, not plain(): inside a code span a backslash would be displayed.
    at = upload(run_app(FakeApi()), "single_upload", ("photo.png", PNG, "image/png"))
    raw = [c.value for c in at.caption]
    assert any("`resnet18-test` · 31 ms inference" in value for value in raw)
    assert not any("resnet18\\-test" in value for value in raw)


def test_missing_metrics_show_a_dash_not_zero(run_app, monkeypatch):
    api = FakeApi()
    real_call = api.__call__

    def without_accuracy(request):
        response = real_call(request)
        if request.url.path == "/model":
            body = response.json()
            del body["evaluation_metrics"]["test"]["accuracy"]
            return httpx.Response(200, json=body)
        return response

    at = run_app(without_accuracy)
    about = {m.label: m.value for m in at.main.metric}
    assert about["Test macro-F1"] == "0.940" and about["Test accuracy"] == "—"


def test_sidebar_and_about_page_layout(run_app):
    at = run_app(FakeApi())
    assert not at.exception
    # Sidebar: service status and the refresh button only (no metrics, model, or connection).
    assert [s.value for s in at.sidebar.subheader] == ["Service status"]
    assert any("model ready" in s.value for s in at.sidebar.success)
    assert [b.label for b in at.sidebar.button] == ["Refresh status"]
    assert not at.sidebar.metric and not at.sidebar.markdown and not at.sidebar.caption
    # About page: all four evaluation metrics, the six classes, no limitations text.
    assert {m.label: m.value for m in at.main.metric} == {
        "Test macro-F1": "0.940", "Test accuracy": "0.939",
        "Validation macro-F1": "0.949", "Test images": "2,997",
    }  # fmt: skip
    cards = [
        m.value
        for m in at.main.markdown
        if m.value.startswith("**") and "fine-tuned" not in m.value
    ]
    assert [plain(c).strip("*").split(" ", 1)[1] for c in cards] == [
        c.capitalize() for c in CLASSES
    ]
    assert any(
        m.value == "**resnet18** fine-tuned for scene classification." for m in at.main.markdown
    )
    rendered = texts(at).lower()
    assert "limitation" not in rendered and "confusion" not in rendered
    # Removed text: version line on the About page and the privacy sentence on every page.
    assert "model version" not in rendered and "training commit" not in rendered
    assert "not stored by this app" not in rendered


def test_code_spans_cannot_be_broken_out_of():
    assert app.code("a`b\nc") == "`a'b c`"


def test_offline_api_is_reported_without_crashing(run_app):
    at = run_app(FakeApi(online=False))
    assert not at.exception
    assert any("Cannot reach the inference API at http://api.test:8000" in plain(e.value)
               for e in at.error)  # fmt: skip


def test_model_not_loaded_blocks_classification(run_app):
    api = FakeApi(ready=False)
    at = upload(run_app(api), "single_upload", ("photo.png", PNG, "image/png"))
    assert not at.exception
    assert any("model not loaded" in w.value for w in at.sidebar.warning)
    assert any("unavailable" in w.value for w in at.warning)
    assert api.calls("/predict") == []


def test_invalid_configuration_is_reported(run_app):
    at = run_app(FakeApi(), url="ftp://api.test")
    assert any("Configuration error" in e.value for e in at.error)


# --- Single image ------------------------------------------------------------------------------


def test_single_prediction_is_displayed_once(run_app, records):
    api = FakeApi()
    at = upload(run_app(api), "single_upload", ("photo.png", PNG, "image/png"))
    assert not at.exception
    html = "\n".join(m.value for m in at.markdown)
    assert 'class="dv-label"' in html and "Forest" in html and "90.0% confidence" in html
    assert at.get("vega_lite_chart"), "the probability chart is rendered"
    assert len(at.get("image")) == 1, "the validated image is previewed"
    assert api.calls("/predict") == [["photo.png"]]
    at.run()  # a rerun (any widget interaction) must not send the image again
    assert len(api.calls("/predict")) == 1
    log = "\n".join(records.lines)
    assert "POST /predict -> 200" in log and "photo.png" not in log


def test_low_confidence_warning(run_app):
    body = prediction("glacier", 0.45) | {"model_version": "v", "inference_ms": 1.0}
    body["top_k"] = [
        {"label": "glacier", "probability": 0.45},
        {"label": "mountain", "probability": 0.11},
    ]
    at = upload(run_app(FakeApi(body=body)), "single_upload", ("photo.jpg", JPEG, "image/jpeg"))
    assert any("Low confidence" in w.value and "mountain" in w.value for w in at.warning)


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("notes.png", b"just some text", "not a PNG image"),
        ("photo.jpg", PNG, "not a JPEG image"),
        ("empty.png", b"", "empty"),
    ],
)
def test_invalid_upload_is_rejected_before_preview_and_sending(run_app, name, data, message):
    api = FakeApi()
    at = upload(run_app(api), "single_upload", (name, data, "image/png"))
    assert any(message in plain(e.value) for e in at.error)
    assert not at.get("image") and api.calls("/predict") == []


def test_file_names_are_escaped(run_app):
    at = upload(run_app(FakeApi()), "single_upload", ("[click](evil)*x*.png", b"x", "image/png"))
    error = at.error[0].value
    assert "\\[click\\]\\(evil\\)\\*x\\*" in error and "](evil)" not in error


@pytest.mark.parametrize("name", ["payload.exe", "no-extension"])
def test_forged_uploads_with_other_extensions_leak_nothing(run_app, name):
    # The browser's file picker only offers .jpg/.jpeg/.png, so only a forged upload message
    # can carry these names. Streamlit rejects them while restoring widget state, before the
    # app runs, so the app cannot intercept them: the page shows Streamlit's generic error
    # with details redacted (showErrorDetails = "none") and nothing is sent to the API.
    api = FakeApi()
    at = upload(run_app(api), "single_upload", (name, PNG, "image/png"))
    assert len(at.exception) == 1
    message = at.exception[0].message
    assert "redacted" in message and name not in message and "extension" not in message
    assert api.calls("/predict") == []


def test_unusual_but_allowed_names_are_validated_by_content(run_app):
    # Names Streamlit accepts reach the app's own checks: content decides, names are escaped.
    api = FakeApi()
    at = upload(run_app(api), "single_upload", ("dir/odd name [1].png", PNG, "image/png"))
    assert not at.exception and api.calls("/predict") == [["dir/odd name [1].png"]]
    assert any("odd name \\[1\\]" in c.value for c in at.caption)


def test_api_errors_are_shown_with_retry(run_app):
    api = FakeApi(predict_status=500)
    at = upload(run_app(api), "single_upload", ("photo.png", PNG, "image/png"))
    assert any("failed to process the request (HTTP 500)" in plain(e.value) for e in at.error)
    retry = next(b for b in at.button if b.label == "Try again")
    retry.click().run()
    assert len(api.calls("/predict")) == 2


def test_malformed_api_response_is_not_displayed(run_app):
    at = upload(run_app(FakeApi(body={"label": "forest", "confidence": 2})), "single_upload",
                ("photo.png", PNG, "image/png"))  # fmt: skip
    assert any("unexpected response" in plain(e.value) for e in at.error)
    assert not at.get("vega_lite_chart")


# --- Batch ------------------------------------------------------------------------------------


def test_batch_with_local_and_api_rejections(run_app, records):
    api = FakeApi(bad_names={"broken.png"})
    files = [("a.png", PNG, "image/png"), ("sea.jpg", JPEG, "image/jpeg"),
             ("notes.png", b"text", "image/png"), ("broken.png", PNG, "image/png"),
             ("b.png", PNG, "image/png")]  # fmt: skip
    at = upload(run_app(api), "batch_upload_0", files)
    assert "4 image(s) ready, 1 rejected before sending" in texts(at)
    next(b for b in at.button if b.label.startswith("Classify 4")).click()
    at.run()
    assert not at.exception

    # The locally rejected file is never sent; the API-rejected one is dropped and the rest resent.
    assert api.calls("/predict/batch") == [["a.png", "sea.jpg", "broken.png", "b.png"],
                                           ["a.png", "sea.jpg", "b.png"]]  # fmt: skip
    metrics = {
        m.label: m.value
        for m in at.metric
        if m.label in ("Images", "Classified", "Failed or rejected")
    }
    assert metrics == {"Images": "5", "Classified": "3", "Failed or rejected": "2"}
    table = at.dataframe[0].value
    assert list(table["File"]) == [f[0] for f in files]
    assert list(table["Status"]) == ["Classified", "Classified", "Rejected", "Failed", "Classified"]
    assert list(table["Prediction"])[:2] == ["Forest", "Sea"]
    assert "The image data is corrupt" in table["Details"][3]
    assert at.get("download_button")
    log = "\n".join(records.lines)
    assert "batch of 4 files: 1 failed, 2 requests" in log and "broken.png" not in log


def test_batch_limit_on_number_of_files(run_app, monkeypatch):
    monkeypatch.setenv("DEEPVISION_FRONTEND_MAX_FILES", "2")
    api = FakeApi()
    at = upload(run_app(api), "batch_upload_0", [(f"{i}.png", PNG, "image/png") for i in range(3)])
    assert any("only the first 2 are used" in w.value for w in at.warning)
    next(b for b in at.button if b.label.startswith("Classify")).click()
    at.run()
    assert api.calls("/predict/batch") == [["0.png", "1.png"]]


def batch_uploader(at: AppTest):
    [uploader] = [u for u in at.file_uploader if u.key.startswith("batch_upload_")]
    return uploader


def test_clear_all_empties_the_batch_and_allows_a_new_one(run_app):
    api = FakeApi()
    at = upload(run_app(api), "batch_upload_0", [("a.png", PNG, "image/png"),
                                                 ("notes.png", b"text", "image/png")])  # fmt: skip
    next(b for b in at.button if b.label.startswith("Classify")).click()
    at.run()
    assert at.dataframe and len(at.get("image")) == 1 and "batch" in at.session_state

    next(b for b in at.button if b.label == "Clear all").click()
    at.run()
    assert not at.exception
    # Everything batch-specific is gone: files, previews, results table, metrics, state.
    assert batch_uploader(at).value in (None, [])
    assert not at.dataframe and not at.get("image") and not at.get("download_button")
    assert {m.label for m in at.main.metric} == {  # only the About page's metrics remain
        "Test macro-F1", "Test accuracy", "Validation macro-F1", "Test images",
    }  # fmt: skip
    assert "batch" not in at.session_state
    assert not [b for b in at.button if b.label in ("Clear all",) or b.label.startswith("Classify")]
    assert any("Upload up to" in i.value for i in at.info)  # back to the empty state
    at.run()  # cleared files do not come back on a rerun
    assert batch_uploader(at).value in (None, []) and not at.dataframe
    assert len(api.calls("/predict/batch")) == 1  # clearing sends nothing

    # A new batch works normally after clearing.
    batch_uploader(at).set_value([("b.png", PNG, "image/png"), ("sea.jpg", JPEG, "image/jpeg")])
    at.run()
    assert "2 image(s) ready" in texts(at)
    next(b for b in at.button if b.label.startswith("Classify 2")).click()
    at.run()
    assert api.calls("/predict/batch")[-1] == ["b.png", "sea.jpg"]
    assert list(at.dataframe[0].value["Status"]) == ["Classified", "Classified"]


def test_clear_all_only_appears_with_files(run_app):
    at = run_app(FakeApi())
    assert not [b for b in at.button if b.label == "Clear all"]


def test_results_csv_neutralizes_formulas():
    rows = [{"name": "=HYPERLINK(1)", "status": "failed", "label": None, "confidence": None,
             "error": "+cmd"}, {"name": "ok.png", "status": "classified", "label": "sea",
             "confidence": 0.5, "error": None}]  # fmt: skip
    text = app.results_csv(rows).decode()
    assert "'=HYPERLINK(1)" in text and "'+cmd" in text and "ok.png,classified,sea,0.5000," in text


def test_markdown_escaping():
    assert (
        app.escape("[x](http://e.test) *b* <i>") == "\\[x\\]\\(http\\://e\\.test\\) \\*b\\* \\<i\\>"
    )
    assert app.escape("a\nb") == "a b" and len(app.escape("x" * 1000)) == 200


def test_importing_the_module_does_not_render_anything():
    # `streamlit run` executes app.py as __main__; importing it only defines helpers.
    assert callable(app.main) and app.ACCENT.startswith("#")


# --- Streamlit configuration ------------------------------------------------------------


def test_streamlit_config_is_hardened():
    config = tomllib.loads((REPO_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["browser"]["gatherUsageStats"] is False  # no usage telemetry
    assert config["logger"]["hideWelcomeMessage"] is True  # no public-IP lookup at startup
    assert config["client"]["showErrorDetails"] == "none"  # no tracebacks in the browser
    assert config["server"]["enableXsrfProtection"] is True
    assert config["server"]["enableStaticServing"] is False
    assert config["server"]["maxUploadSize"] == 10  # MB, same as the API default
    assert config["theme"]["base"] == "dark"
