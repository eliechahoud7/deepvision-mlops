"""DeepVision Streamlit frontend.

    streamlit run src/deepvision/frontend/app.py

A client of the inference API (DEEPVISION_API_URL, default http://127.0.0.1:8000): it
never loads the model. Uploads are validated (type, signature, size, pixel count) before
they are previewed or sent, the API's responses are validated before they are shown, and
user-controlled text (file names, error messages) is escaped before it is rendered.
Logs carry only counts, status codes, and timings, never image bytes or file names.
"""

import csv
import hashlib
import io
import logging

import altair as alt
import pandas as pd
import streamlit as st

from deepvision.frontend import client as api
from deepvision.frontend.uploads import ACCEPTED_TYPES, UploadError, ValidatedImage, validate_upload

logger = logging.getLogger("deepvision.frontend")

ACCENT = "#4F8BF9"
MUTED = "#3A4256"
LOW_CONFIDENCE = 0.6
ICONS = {"buildings": "🏙️", "forest": "🌲", "glacier": "🧊", "mountain": "⛰️", "sea": "🌊",
         "street": "🛣️"}  # fmt: skip
MARKDOWN_SPECIAL = "\\`*_{}[]()<>#+-.!|~$:"
CSS = """
<style>
.dv-label {font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0.1rem 0 0.2rem;}
.dv-kicker {font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; opacity: 0.65;}
.dv-confidence {font-size: 1rem; opacity: 0.85;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""


def escape(text: str) -> str:
    """Render user-controlled text literally in Streamlit markdown."""
    text = str(text).replace("\n", " ")[:200]
    return "".join(f"\\{c}" if c in MARKDOWN_SPECIAL else c for c in text)


def code(text: str) -> str:
    """Inline code span. Markdown ignores backslash escapes inside code, so `escape` would
    show its backslashes there; instead, backticks (the only way out of the span) are
    replaced."""
    return "`" + str(text).replace("`", "'").replace("\n", " ")[:100] + "`"


def fmt_metric(metrics: dict[str, float] | None, key: str) -> str:
    return f"{metrics[key]:.3f}" if metrics and key in metrics else "—"


def configure_logging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    for name in ("httpx", "httpcore"):  # their INFO lines would duplicate ours
        logging.getLogger(name).setLevel(logging.WARNING)


# --- API access (cached so Streamlit reruns do not flood the API) ---------------------------


@st.cache_resource(show_spinner=False)
def get_client(settings: api.FrontendSettings) -> api.ApiClient:
    return api.ApiClient(settings)


@st.cache_data(ttl=10, show_spinner=False)
def fetch_status(settings: api.FrontendSettings) -> api.ServiceStatus:
    return get_client(settings).status()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_info(settings: api.FrontendSettings) -> tuple[api.ModelInfo | None, str | None]:
    try:
        return get_client(settings).model_info(), None
    except api.ApiError as exc:
        return None, exc.message


def digest(*chunks: bytes) -> str:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(hashlib.sha256(chunk).digest())
    return hasher.hexdigest()


def to_upload(image: ValidatedImage) -> api.Upload:
    return api.Upload(image.name, image.data, image.content_type)


# --- Presentation helpers ----------------------------------------------------------------


def title(label: str) -> str:
    return f"{ICONS.get(label, '')} {label.capitalize()}".strip()


def probability_chart(probabilities: dict[str, float], predicted: str) -> alt.Chart:
    data = pd.DataFrame(
        {
            "class": [name.capitalize() for name in probabilities],
            "probability": list(probabilities.values()),
            "predicted": [name == predicted for name in probabilities],
        }
    )
    base = alt.Chart(data).encode(
        y=alt.Y("class:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=13, ticks=False)),
        x=alt.X(
            "probability:Q",
            title=None,
            scale=alt.Scale(domain=[0, 1.15]),
            axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], grid=False),
        ),
        tooltip=[alt.Tooltip("class:N"), alt.Tooltip("probability:Q", format=".2%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=20).encode(
        color=alt.condition("datum.predicted", alt.value(ACCENT), alt.value(MUTED))
    )
    labels = base.mark_text(align="left", dx=6, fontSize=12).encode(
        text=alt.Text("probability:Q", format=".1%")
    )
    return (bars + labels).properties(height=len(data) * 38).configure_view(stroke=None)


def distribution_chart(labels: list[str]) -> alt.Chart:
    counts = pd.Series(labels).value_counts().reindex(api.CLASS_NAMES, fill_value=0)
    data = pd.DataFrame({"class": [c.capitalize() for c in counts.index], "images": counts.values})
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=ACCENT)
        .encode(
            x=alt.X("class:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("images:Q", title=None, axis=alt.Axis(tickMinStep=1)),
            tooltip=["class:N", "images:Q"],
        )
        .properties(height=220)
        .configure_view(stroke=None)
    )


def csv_cell(value) -> str:
    """Neutralize spreadsheet formulas in user-controlled cells (CSV injection)."""
    text = "" if value is None else str(value)
    return f"'{text}" if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def results_csv(rows: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["file", "status", "prediction", "confidence", "error"])
    for row in rows:
        confidence = f"{row['confidence']:.4f}" if row["confidence"] is not None else ""
        writer.writerow([csv_cell(row["name"]), row["status"], row["label"] or "", confidence,
                         csv_cell(row["error"])])  # fmt: skip
    return buffer.getvalue().encode("utf-8")


# --- Page sections ------------------------------------------------------------------------


def render_sidebar(status: api.ServiceStatus) -> None:
    with st.sidebar:
        st.subheader("Service status")
        if status.ready:
            st.success("API online · model ready", icon=":material/check_circle:")
        elif status.online:
            st.warning(f"API online · model not loaded. {escape(status.message)}",
                       icon=":material/warning:")  # fmt: skip
        else:
            st.error(escape(status.message), icon=":material/cloud_off:")
        if st.button("Refresh status", icon=":material/refresh:", width="stretch"):
            fetch_status.clear()
            fetch_model_info.clear()
            st.rerun()


def render_prediction(result: api.PredictionResult) -> None:
    with st.container(border=True):
        st.markdown(
            f'<div class="dv-kicker">Predicted scene</div>'
            f'<div class="dv-label">{title(result.label)}</div>'
            f'<div class="dv-confidence">{result.confidence:.1%} confidence</div>',
            unsafe_allow_html=True,  # only validated labels and numbers are interpolated
        )
        if result.confidence < LOW_CONFIDENCE:
            runner_up = result.top_k[1][0] if len(result.top_k) > 1 else None
            hint = f" It may also be **{runner_up}**." if runner_up else ""
            st.warning(f"Low confidence: the model is unsure about this image.{hint}",
                       icon=":material/help:")  # fmt: skip
        st.altair_chart(probability_chart(result.probabilities, result.label), width="stretch")
        st.caption(f"Model {code(result.model_version)} · {result.inference_ms:.0f} ms inference")


def render_single(settings: api.FrontendSettings, status: api.ServiceStatus) -> None:
    uploaded = st.file_uploader(
        "Upload a scene photo",
        type=list(ACCEPTED_TYPES),
        key="single_upload",
        max_upload_size=int(settings.max_upload_mb),
        help="Buildings, forest, glacier, mountain, sea, or street.",
    )
    if uploaded is None:
        st.info("Upload a JPG or PNG photo to classify it into one of the six scene classes.",
                icon=":material/upload:")  # fmt: skip
        return
    data = uploaded.getvalue()
    try:
        image = validate_upload(uploaded.name, data, max_bytes=settings.max_upload_bytes,
                                max_pixels=settings.max_pixels)  # fmt: skip
    except UploadError as exc:
        st.error(f"**{escape(uploaded.name)}**: {escape(exc)}", icon=":material/block:")
        return

    preview, results = st.columns([1, 1.15], gap="large")
    with preview:
        st.image(image.data, width="stretch")
        st.caption(f"{escape(image.name)} · {image.width} × {image.height} px")
    with results:
        if not status.ready:
            st.warning("Classification is unavailable until the API reports the model as ready.",
                       icon=":material/hourglass_empty:")  # fmt: skip
            return
        key = digest(image.data)
        cached = st.session_state.get("single")
        if cached is None or cached[0] != key:
            with st.spinner("Classifying…"):
                try:
                    cached = (key, get_client(settings).predict(to_upload(image)), None)
                except api.ApiError as exc:
                    cached = (key, None, exc.message)
            st.session_state["single"] = cached
        _, result, error = cached
        if error:
            st.error(escape(error), icon=":material/error:")
            if st.button("Try again", icon=":material/refresh:"):
                del st.session_state["single"]
                st.rerun()
            return
        render_prediction(result)


def clear_batch() -> None:
    """Clear All: a file uploader cannot be emptied through session state, so it gets a new
    key (Streamlit discards the old widget and its files) and the batch results are dropped.
    Runs as an on_click callback, before the rerun, so cleared files never render again."""
    st.session_state["batch_uploader"] = st.session_state.get("batch_uploader", 0) + 1
    st.session_state.pop("batch", None)


def render_batch(settings: api.FrontendSettings, status: api.ServiceStatus) -> None:
    files = st.file_uploader(
        "Upload several scene photos",
        type=list(ACCEPTED_TYPES),
        accept_multiple_files=True,
        key=f"batch_upload_{st.session_state.get('batch_uploader', 0)}",
        max_upload_size=int(settings.max_upload_mb),
    )
    if not files:
        st.info(f"Upload up to {settings.max_files} images to classify them together. "
                "Invalid images are reported individually and do not stop the others.",
                icon=":material/photo_library:")  # fmt: skip
        return
    if len(files) > settings.max_files:
        st.warning(f"{len(files)} files uploaded; only the first {settings.max_files} are used.")
        files = files[: settings.max_files]

    rows: list[dict] = []
    valid: list[tuple[int, ValidatedImage]] = []
    for position, file in enumerate(files):
        row = {"name": file.name, "status": "rejected", "label": None, "confidence": None,
               "error": None, "image": None}  # fmt: skip
        try:
            row["image"] = validate_upload(file.name, file.getvalue(),
                                           max_bytes=settings.max_upload_bytes,
                                           max_pixels=settings.max_pixels)  # fmt: skip
            row["status"] = "ready"
            valid.append((position, row["image"]))
        except UploadError as exc:
            row["error"] = str(exc)
        rows.append(row)

    rejected = len(rows) - len(valid)
    st.caption(
        f"{len(valid)} image(s) ready"
        + (f", {rejected} rejected before sending" if rejected else "")
    )
    key = digest(*(image.data for _, image in valid), *(r["name"].encode() for r in rows))
    with st.container(horizontal=True):
        clicked = st.button(
            f"Classify {len(valid)} image{'s' if len(valid) != 1 else ''}",
            type="primary",
            icon=":material/bolt:",
            disabled=not valid or not status.ready,
        )
        st.button("Clear all", icon=":material/delete_sweep:", on_click=clear_batch)
    if not status.ready:
        st.caption("Classification is unavailable until the API reports the model as ready.")
    if clicked:
        with st.spinner(f"Classifying {len(valid)} images…"):
            outcome = get_client(settings).predict_batch([to_upload(image) for _, image in valid])
        st.session_state["batch"] = (key, outcome)
    stored = st.session_state.get("batch")
    if stored is None or stored[0] != key:
        if rejected:
            render_batch_table(rows)
        return

    outcome: api.BatchResult = stored[1]
    for (position, _), item in zip(valid, outcome.items, strict=True):
        row = rows[position]
        if item.ok:
            row.update(status="classified", label=item.prediction.label,
                       confidence=item.prediction.confidence)  # fmt: skip
        else:
            row.update(status="failed", error=item.error)
    classified = [row for row in rows if row["status"] == "classified"]

    columns = st.columns(4)
    columns[0].metric("Images", len(rows))
    columns[1].metric("Classified", len(classified))
    columns[2].metric("Failed or rejected", len(rows) - len(classified))
    mean = sum(r["confidence"] for r in classified) / len(classified) if classified else None
    columns[3].metric("Mean confidence", f"{mean:.1%}" if mean is not None else "—")

    if classified:
        with st.container(border=True):
            st.markdown("**Predicted classes**")
            st.altair_chart(distribution_chart([r["label"] for r in classified]), width="stretch")
    render_batch_table(rows)
    st.download_button("Download results (CSV)", results_csv(rows), "deepvision-batch.csv",
                       "text/csv", icon=":material/download:")  # fmt: skip
    render_gallery(rows)


def render_batch_table(rows: list[dict]) -> None:
    table = pd.DataFrame(
        {
            "File": [r["name"] for r in rows],
            "Prediction": [r["label"].capitalize() if r["label"] else "—" for r in rows],
            "Confidence": [r["confidence"] for r in rows],
            "Status": [r["status"].capitalize() for r in rows],
            "Details": [r["error"] or "" for r in rows],
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", format="percent", min_value=0.0, max_value=1.0
            ),
        },
    )


def render_gallery(rows: list[dict], per_row: int = 4) -> None:
    st.markdown("**Images**")
    for start in range(0, len(rows), per_row):
        for column, row in zip(st.columns(per_row), rows[start : start + per_row], strict=False):
            with column, st.container(border=True):
                if row["image"] is not None:
                    st.image(row["image"].data, width="stretch")
                if row["status"] == "classified":
                    st.markdown(f"**{title(row['label'])}** · {row['confidence']:.1%}")
                else:
                    st.markdown(
                        f":red[**{row['status'].capitalize()}**] · {escape(row['error'] or '')}"
                    )
                st.caption(escape(row["name"]))


def render_about(settings: api.FrontendSettings, status: api.ServiceStatus) -> None:
    if not status.ready:
        st.info("Model details appear once the API reports the model as ready.")
        return
    info, error = fetch_model_info(settings)
    if error or info is None:
        st.error(escape(error or "Model details are unavailable."))
        return
    st.markdown(f"**{escape(info.backbone)}** fine-tuned for scene classification.")
    metrics = st.columns(4)
    test = info.test_metrics or {}
    metrics[0].metric("Test macro-F1", fmt_metric(test, "macro_f1"))
    metrics[1].metric("Test accuracy", fmt_metric(test, "accuracy"))
    metrics[2].metric("Validation macro-F1", fmt_metric(info.validation_metrics, "macro_f1"))
    metrics[3].metric("Test images", f"{int(test['images']):,}" if "images" in test else "—")
    st.caption("SUPPORTED CLASSES")
    for start in range(0, len(info.class_names), 3):
        for column, name in zip(st.columns(3), info.class_names[start : start + 3], strict=False):
            with column, st.container(border=True):
                st.markdown(f"**{escape(title(name))}**")  # names come from /model


def main() -> None:
    st.set_page_config(page_title="DeepVision · Scene classification", page_icon="🏔️",
                       layout="wide", initial_sidebar_state="expanded")  # fmt: skip
    configure_logging()
    st.markdown(CSS, unsafe_allow_html=True)  # static styles only
    try:
        settings = api.FrontendSettings.from_env()
    except api.SettingsError as exc:
        st.error(f"Configuration error: {escape(exc)}")
        st.stop()

    status = fetch_status(settings)
    render_sidebar(status)
    st.title("DeepVision")
    st.markdown("Scene classification with a fine-tuned ResNet-18, served by the DeepVision "
                "inference API.")  # fmt: skip
    single, batch, about = st.tabs(["Single image", "Batch", "About the model"])
    with single:
        render_single(settings, status)
    with batch:
        render_batch(settings, status)
    with about:
        render_about(settings, status)


if __name__ == "__main__":  # `streamlit run` executes this file as __main__
    main()
