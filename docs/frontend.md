# Streamlit frontend

A dark-themed web interface for the DeepVision inference API. It runs in its own
container and **never loads the model**: every prediction is an HTTP request to the
FastAPI service, which remains the only component that runs the network.

## Using it

| Area | What it shows |
|---|---|
| Sidebar | API status (*online · model ready*, *online · model not loaded*, or *offline*) and a refresh button |
| **Single image** | Upload a JPG/JPEG/PNG, see a preview and the predicted scene with its confidence, a low-confidence note when the model is unsure, and a chart of all six class probabilities |
| **Batch** | Upload up to 64 images, then *Classify*: a summary (classified, failed, mean confidence), a chart of predicted classes, a results table, a CSV download, and a thumbnail gallery. Invalid images are reported one by one and never stop the others. *Clear all* removes the whole batch and its results at once. |
| **About the model** | Backbone; held-out test macro-F1, test accuracy, validation macro-F1, and test image count; the six supported classes |

## Running

With Docker Compose (Windows PowerShell, repository root; the checkpoint must exist as
for the API):

```powershell
docker compose up --build -d          # api, frontend, prometheus, grafana
Start-Process http://127.0.0.1:8501   # the frontend
```

Without Docker, start the API (see `docs/api.md`) and then, with the virtual environment
active:

```powershell
streamlit run src/deepvision/frontend/app.py   # reads .streamlit/config.toml from the root
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `DEEPVISION_API_URL` | `http://127.0.0.1:8000` (Compose: `http://api:8000`) | API base URL; `http`/`https` only, no credentials, query, or fragment |
| `DEEPVISION_MAX_UPLOAD_MB` | `10` | Maximum size of one image (shared with the API in Compose) |
| `DEEPVISION_MAX_BATCH` | `16` | Images per `/predict/batch` request (shared with the API in Compose) |
| `DEEPVISION_FRONTEND_MAX_FILES` | `64` | Images per batch upload in the UI, sent in chunks of `DEEPVISION_MAX_BATCH` |
| `FRONTEND_PORT` (Compose) | `8501` | Host port, always bound to `127.0.0.1` |

In Compose, the frontend reads the same `DEEPVISION_MAX_UPLOAD_MB` and
`DEEPVISION_MAX_BATCH` as the API, so the limits cannot drift apart. Streamlit's own
upload limit (`server.maxUploadSize`) is set from the same variable and must be a whole
number of megabytes.

## How it talks to the API

- **Server-side only.** The Streamlit server calls the API with `httpx`; the browser never
  does. In Compose the two share a private `frontend` network.
- **Timeouts:** 3 s to connect, 30 s to read (60 s for batches). Redirects are not followed.
- **Caching:** `/health` + `/ready` for 10 s and `/model` for 5 minutes, so Streamlit's
  reruns do not flood the API (or its Prometheus counters). A single-image prediction is
  kept for the uploaded file, so interacting with the page never resends it.
- **Validated responses.** Labels must be one of the six classes; probabilities must be
  finite, within [0, 1], cover exactly the six classes, and sum to 1; the confidence must
  be the top probability; batch results must match the files sent, in order. Anything
  else is shown as "The API returned an unexpected response", never displayed raw.
- **Batch isolation.** `/predict/batch` rejects the whole request when one file is bad
  (`"file 3: …"`). The frontend marks that file as failed with the API's reason and
  resends the rest. Every retry removes a file, so a batch of *n* images needs at most
  *n* requests; a second cap enforces the same bound.

## Security

- **Uploads are validated before they are previewed or sent:** extension (JPG, JPEG, PNG),
  non-empty, size limit, file signature matching the extension (a renamed file is
  rejected), and the pixel count read from the header without decoding the image, which
  rules out decompression bombs. The API repeats its own checks.
- **No telemetry or outside requests:** `browser.gatherUsageStats = false`, and
  `logger.hideWelcomeMessage = true` skips the public-IP lookup that Streamlit otherwise
  makes for its startup banner. Fonts and assets are bundled.
- **No leaked details:** `client.showErrorDetails = "none"` hides exceptions from the
  browser; error messages are short and path-free; file names and messages are escaped
  before rendering; CSV cells that could start a spreadsheet formula are neutralized.
- **Logs** contain request paths, status codes, counts, and timings, never image bytes or
  file names.
- **Container:** non-root (UID 10001), read-only root filesystem with a small `noexec`
  `/tmp`, all capabilities dropped, `no-new-privileges`, memory/PID/log limits, port on
  `127.0.0.1` only. The image contains only `deepvision.frontend` and its dependencies (no
  PyTorch, no model code, no checkpoint) and is isolated from Prometheus and Grafana.
- **No secrets.** The frontend needs none; `.streamlit/secrets.toml` is ignored by Git and
  excluded from the Docker build context.

## Tests

`tests/unit/test_frontend_client.py` (API client with a mock transport: responses,
errors, timeouts, malformed data, batch isolation and its bound),
`test_frontend_uploads.py` (validation, including decompression bombs without decoding),
and `test_frontend_app.py` (the real app with Streamlit's `AppTest` against a fake API).
CI also builds the frontend image and smoke-tests it alone and in the Compose stack.

## Limitations

- **No authentication.** Like the API, the frontend is for local use; keep it on
  localhost or put it behind authentication and TLS.
- **Forged uploads.** The browser's file picker only offers JPG, JPEG, and PNG files, but a
  crafted upload message can carry another extension (or none). Streamlit rejects such a
  file in its own type check before the app runs, so the app cannot show a friendlier
  message: the page shows Streamlit's generic error (details redacted) until it is
  reloaded, and nothing reaches the API. Names Streamlit accepts are always checked by
  content, so a misleading name alone gets nothing through.
- **The batch limit is per upload**, not per user: concurrent users each send their own
  requests.
- Predictions are only as good as the model: every image gets one of the six labels.
