# WebApp — dashboard, REST API, deployment

The dashboard (`app/`) and the REST API (`api/`) are two thin front-ends over
one scoring core (`core/`). They import `src/` from `../Maternal_Health/`,
which is where the bundles' classes are defined. They
can run together in one container or as two separate services; either way
they need the trained bundles and, if you want dashboard-issued API keys to
work against the API, one shared secret.

```
┌────────────────────────── one container (Hugging Face Space / any Docker host) ─┐
│  nginx :7860 ──►  /api/*  ──►  uvicorn :8000   WebApp/api      ┐               │
│             └──►  /*      ──►  streamlit :8501  WebApp/app      ┴► core.engine   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Layout

```
WebApp/
├── streamlit_app.py      dashboard entry point (Community Cloud main file)
├── requirements.txt      pinned runtime set, CPU torch
├── app/                  Streamlit pages, theme, components
├── api/main.py           FastAPI service
├── core/                 engine, model store, API keys, presets, labels, paths
├── deploy/               nginx.conf + start.sh for the one-container build
├── scripts/              upload_models.py
└── test_serving.py
```

Three files must stay at the repository root because the platforms look for
them there: `Dockerfile` (Hugging Face Spaces), `packages.txt` (Community
Cloud apt packages) and `.streamlit/config.toml` (theme, read from the CWD).

## 0. Publish the model bundles (once)

The bundles are not in git — `neonatal_death.joblib` alone is 133 MB. Put them
on the Hugging Face Hub from the machine where `train.py` wrote them:

```bash
pip install huggingface_hub
huggingface-cli login                                   # or: export HF_TOKEN=hf_...
python WebApp/scripts/upload_models.py FarhanKO/mch-risk-models   # add --private if you like
```

Every deployment below then just needs `MCH_MODEL_REPO=FarhanKO/mch-risk-models`
(and `HF_TOKEN` if the repo is private). The bundles are downloaded on first
start (≈215 MB, a few seconds from the Hub) and cached in `Maternal_Health/models/`.

Alternatives, in the order the app checks them: a directory that already holds
the bundles (`MCH_MODELS_DIR`), the Hub repo above, or a JSON map of direct
download URLs in `MCH_MODEL_URLS`.

## 1. Hugging Face Spaces — dashboard + API on one URL (recommended)

Free tier: 2 vCPU, 16 GB RAM, which is comfortable for torch + four boosting
libraries + the bundles. This is the only free host that runs both the UI and
a real REST endpoint from one repo.

1. Create a Space → **Docker** SDK → blank template.
2. Push this repository to it (the root `README.md` front matter already declares
   `sdk: docker` and `app_port: 7860`; the `Dockerfile` at the root builds from `WebApp/`):
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-user>/<space-name>
   git push space main
   ```
3. In the Space's **Settings → Variables and secrets** add:

   | name | value | kind |
   |---|---|---|
   | `MCH_MODEL_REPO` | `FarhanKO/mch-risk-models` | variable |
   | `MCH_API_SECRET` | a long random string (`python -c "import secrets;print(secrets.token_urlsafe(48))"`) | **secret** |
   | `HF_TOKEN` | only if the model repo is private | secret |

4. Wait for the build (≈5–8 min the first time). The dashboard is at
   `https://<hf-user>-<space-name>.hf.space/`, the API at `…/api/`, the OpenAPI
   docs at `…/api/docs`. The API page in the dashboard detects this layout and
   fills the base URL in automatically.

`WebApp/deploy/start.sh` generates a random `MCH_API_SECRET` if you skip it, but keys
then stop working on every restart — set it.

## 2. Streamlit Community Cloud — dashboard only

Community Cloud runs one Streamlit process and cannot expose a second port,
so the REST API has to live elsewhere (option 1 or 3). The dashboard itself
works fully, including in-process scoring and key generation.

1. **New app** → this repo → branch `main` → main file **`WebApp/streamlit_app.py`**.
2. **Advanced settings → Python version**: 3.13 (the bundles were pickled with
   pandas 3 / scikit-learn 1.8, which need 3.11+).
3. **Secrets** (TOML):
   ```toml
   MCH_MODEL_REPO   = "FarhanKO/mch-risk-models"
   MCH_API_SECRET   = "<the same secret the API uses>"
   MCH_API_BASE_URL = "https://<hf-user>-<space-name>.hf.space/api"   # optional: link a deployed API
   # HF_TOKEN       = "hf_..."                                          # only for a private model repo
   ```
4. Deploy. First boot installs `WebApp/requirements.txt` (found because it sits
   beside the entrypoint; CPU torch from the PyTorch index — the
   `--extra-index-url` line matters), `packages.txt` from the root, and pulls
   the bundles.

Resource note: the process sits around 1 GB RSS with every bundle loaded,
inside Community Cloud's limit, but leave the app on 3.13 and do not add the
training-only packages from `Maternal_Health/requirements.txt`.

## 3. Any Docker host — Render, Railway, Fly.io, a VM

Same image as the Space:

```bash
docker build -t mch .
docker run -p 7860:7860 \
  -e MCH_MODEL_REPO=FarhanKO/mch-risk-models \
  -e MCH_API_SECRET=<secret> \
  mch
```

Dashboard on `http://localhost:7860/`, API on `http://localhost:7860/api/`.
Give the container ≥ 2 GB RAM. To run **only** the API (for example next to a
Community Cloud dashboard):

```bash
docker run -p 8000:8000 -e MCH_MODEL_REPO=… -e MCH_API_SECRET=… mch \
  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --app-dir WebApp
```

## Configuration reference

| variable | used by | meaning | default |
|---|---|---|---|
| `MCH_MODEL_REPO` | both | Hugging Face repo id holding the bundles | — |
| `MCH_MODEL_URLS` | both | JSON `{ "stunted.joblib": "https://…", … }` | — |
| `MCH_MODELS_DIR` | both | local bundle directory | `<repo>/Maternal_Health/models` |
| `HF_TOKEN` | both | Hub token for a private model repo | — |
| `MCH_API_SECRET` | both | HMAC secret shared by key issuer and verifier | random per process |
| `MCH_API_BASE_URL` | dashboard | public URL of the API, shown on the API page | auto when proxied |
| `MCH_API_INTERNAL_URL` | dashboard | where the dashboard server reaches the API for the playground | `MCH_API_BASE_URL` |
| `MCH_API_REQUIRE_KEY` | API | `false` opens the prediction endpoints | `true` |
| `MCH_RATE_LIMIT` | API | requests per minute per key | `60` |
| `MCH_MAX_BATCH` | API | records per batch call | `500` |
| `MCH_REVOKED_KEYS` | API | comma-separated key ids to refuse | — |
| `MCH_API_ROOT_PATH` | API | mount prefix behind a proxy (`/api`) | — |

On Streamlit, any of these can be given in `.streamlit/secrets.toml` instead
of the environment; the app copies them across at startup.

## Local development

```bash
pip install -r WebApp/requirements.txt
# put the bundles in Maternal_Health/models/ or export MCH_MODEL_REPO=…

# dashboard (from the repo root, so .streamlit/config.toml applies)
streamlit run WebApp/streamlit_app.py

# API, in another terminal
MCH_API_SECRET=dev uvicorn api.main:app --app-dir WebApp --reload --port 8000
```

To link them locally, add `MCH_API_SECRET = "dev"` and
`MCH_API_BASE_URL = "http://localhost:8000"` to `.streamlit/secrets.toml`.

Tests: `cd WebApp && pytest test_serving.py -q` (plus `cd Maternal_Health && python test.py`
for the research package) — the serving tests skip themselves when the bundles are absent.

## How API keys work

`core/apikeys.py`. A key is `mch_<payload>.<hmac-sha256>` where the payload
carries owner, scope, issue and expiry times. The dashboard signs with
`MCH_API_SECRET`; the API recomputes the signature with the same secret.
Nothing is stored, so:

* keys survive restarts and redeploys as long as the secret does;
* two services recognise each other's keys iff they share the secret;
* rotating the secret invalidates every key at once;
* one key is revoked by adding its 8-character id to `MCH_REVOKED_KEYS`.

## Troubleshooting

* **"Model bundles are not available"** — no source configured. Set
  `MCH_MODEL_REPO` (and `HF_TOKEN` for a private repo) or mount the files.
* **Build pulls gigabytes of NVIDIA packages** — the `--extra-index-url` line
  in `WebApp/requirements.txt` was dropped; the PyPI torch wheel bundles CUDA.
* **Keys issued on the dashboard get 401 from the API** — the two processes
  have different `MCH_API_SECRET` values (or one has none and generated its own).
* **Container exits on start** — `WebApp/deploy/start.sh` must have LF line endings
  (`.gitattributes` enforces this) and nginx needs `/tmp/nginx` writable.
* **Pickle / version errors on load** — keep the exact pins in
  `requirements.txt`; the bundles were written under those versions.
