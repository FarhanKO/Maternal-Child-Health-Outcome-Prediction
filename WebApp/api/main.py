"""Maternal & Child Health Risk API.

    uvicorn api.main:app --host 0.0.0.0 --port 8000        # from WebApp/

Endpoints
    GET  /health                 liveness + model version        (open)
    GET  /v1/models              the five bundles, summarised    (open)
    GET  /v1/schema              every accepted field            (open)
    GET  /v1/presets             the four illustrative profiles  (open)
    GET  /v1/keys/verify         inspect the calling key         (key)
    POST /v1/predict             one record -> full cascade      (key)
    POST /v1/predict/batch       up to MCH_MAX_BATCH records     (key)

Authentication is an API key in the `X-API-Key` header (or `Authorization:
Bearer <key>`). Keys are issued by the dashboard's API page and verified
statelessly — see core/apikeys.py. Set MCH_API_REQUIRE_KEY=false to open the
prediction endpoints entirely.

Environment
    MCH_API_SECRET        shared with the dashboard; random per process if unset
    MCH_API_REQUIRE_KEY   default true
    MCH_RATE_LIMIT        requests per minute per key, default 60
    MCH_MAX_BATCH         default 500
    MCH_API_ROOT_PATH     e.g. "/api" when mounted behind a reverse proxy
    MCH_MODEL_REPO / MCH_MODEL_URLS / MCH_MODELS_DIR   see core/model_store.py
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

WEBAPP_DIR = Path(__file__).resolve().parent.parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
import core  # noqa: E402,F401  (puts Maternal_Health/ on sys.path)

from core.apikeys import InvalidKey, KeyInfo, resolve_secret, verify_key  # noqa: E402
from core.engine import CascadeEngine  # noqa: E402
from core.model_store import ModelsUnavailable, ensure_models  # noqa: E402
from core.presets import PRESETS  # noqa: E402

log = logging.getLogger("mch.api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

API_VERSION = "1.0.0"
REQUIRE_KEY = os.environ.get("MCH_API_REQUIRE_KEY", "true").lower() not in ("0", "false", "no")
RATE_LIMIT = int(os.environ.get("MCH_RATE_LIMIT", "60"))
MAX_BATCH = int(os.environ.get("MCH_MAX_BATCH", "500"))
ROOT_PATH = os.environ.get("MCH_API_ROOT_PATH", "").rstrip("/")

DESCRIPTION = """
Population-level risk estimates for five maternal and child health outcomes —
**where she delivers → whether the baby survives → how the child grows** —
from socio-demographic survey inputs, trained on the Bangladesh DHS 2017-18
and 2022 rounds (121,067 records, 45,458 mothers).

Send any subset of the fields listed at `/v1/schema`; whatever you omit is
filled from a cohort-typical template and reported back under
`inputs.defaulted`. Every response carries the operating threshold, the
national rate and a conformal prediction set for each outcome, and a triage
band (`ROUTINE` / `MONITOR` / `PRIORITY`) counted over the four adverse
outcomes.

**Not a diagnosis.** A probability of 0.31 for stunting means roughly a third
of women with this profile had a stunted child — not that this one will.
Probabilities are not calibrated for settings other than the survey rounds
they were trained on.
"""

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
STATE: dict[str, Any] = {"engine": None, "error": None, "secret": None,
                         "started": time.time()}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not os.environ.get("MCH_API_SECRET", "").strip():
        log.warning("MCH_API_SECRET is not set: using a random per-process "
                    "secret, so keys will only validate against this process "
                    "and will not survive a restart.")
    STATE["secret"] = resolve_secret()
    try:
        directory = ensure_models(progress=log.info)
        STATE["engine"] = CascadeEngine(directory)
        log.info("engine ready: version %s, %d targets, %d fields",
                 STATE["engine"].version, len(STATE["engine"].targets),
                 len(STATE["engine"].features))
    except ModelsUnavailable as exc:
        STATE["error"] = str(exc)
        log.error("models unavailable: %s", exc)
    yield


app = FastAPI(
    title="Maternal & Child Health Risk API",
    version=API_VERSION,
    description=DESCRIPTION,
    root_path=ROOT_PATH,
    lifespan=lifespan,
    contact={"name": "Project repository",
             "url": "https://github.com/FarhanKO/Maternal-Child-Health-Outcome-Prediction"},
    license_info={"name": "Code: MIT · Data: DHS Program terms"},
    openapi_tags=[
        {"name": "status", "description": "Liveness and metadata. No key needed."},
        {"name": "reference", "description": "Schema, models and presets. No key needed."},
        {"name": "predict", "description": "Scoring endpoints. API key required."},
    ],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


# ---------------------------------------------------------------------------
# Auth + rate limiting
# ---------------------------------------------------------------------------
_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)

_windows: dict[str, deque] = defaultdict(deque)
_windows_lock = threading.Lock()


def _rate_limit(key_id: str) -> tuple[int, int]:
    """Sliding one-minute window per key. Returns (remaining, reset_in_s)."""
    now = time.time()
    with _windows_lock:
        q = _windows[key_id]
        while q and q[0] <= now - 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return 0, int(q[0] + 60 - now) + 1
        q.append(now)
        return RATE_LIMIT - len(q), 60


def require_key(
    request: Request,
    header_key: str | None = Depends(_header_scheme),
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> KeyInfo:
    raw = header_key or (bearer.credentials if bearer else None)
    if not REQUIRE_KEY and not raw:
        info = KeyInfo(key_id="anonymous", owner="anonymous", scope="predict",
                       issued_at=0, expires_at=None)
    else:
        if not raw:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Send it as `X-API-Key: <key>` or "
                       "`Authorization: Bearer <key>`. Keys are issued on the "
                       "dashboard's API page.",
                headers={"WWW-Authenticate": "ApiKey"})
        try:
            info = verify_key(STATE["secret"], raw)
        except InvalidKey as exc:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid API key ({exc.reason}).",
                headers={"WWW-Authenticate": "ApiKey"}) from exc
    remaining, reset = _rate_limit(info.key_id)
    request.state.rate = (remaining, reset)
    if remaining <= 0:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {RATE_LIMIT} requests/minute exceeded. "
                   f"Retry in {reset}s.",
            headers={"Retry-After": str(reset)})
    return info


def engine() -> CascadeEngine:
    if STATE["engine"] is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=STATE["error"] or "Models are still loading. Retry shortly.")
    return STATE["engine"]


@app.middleware("http")
async def rate_headers(request: Request, call_next):
    response = await call_next(request)
    rate = getattr(request.state, "rate", None)
    if rate:
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT)
        response.headers["X-RateLimit-Remaining"] = str(rate[0])
        response.headers["X-RateLimit-Reset"] = str(rate[1])
    response.headers["X-Model-Version"] = (
        STATE["engine"].version if STATE["engine"] else "unavailable")
    return response


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.detail,
                                 "status": exc.status_code},
                        headers=exc.headers)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
Record = dict[str, str | float | int | bool | None]

EXAMPLE_RECORD = {
    "v012": 17, "v106": "no education", "v190": "poorest", "v025": "rural",
    "v024": "sylhet", "height_cm": 143, "weight_kg": 35.0, "m14": 1,
    "v201": 1, "v511": 15, "risk_index": 3, "b19": 8, "p20": 8,
}


class PredictRequest(BaseModel):
    record: Record = Field(
        ..., description="Any subset of the fields in GET /v1/schema. "
                         "Omitted fields are defaulted; null means unknown.",
        json_schema_extra={"example": EXAMPLE_RECORD})
    fill: Literal["template", "none"] = Field(
        "template", description="`template` fills omitted fields from the "
                                "cohort-typical template; `none` leaves them "
                                "missing.")
    include_anomaly: bool = Field(
        True, description="Score the record against the reference cohort's "
                          "feature distribution.")
    include_record: bool = Field(
        False, description="Echo back the fully-prepared record that was "
                           "scored.")


class BatchRequest(BaseModel):
    records: list[Record] = Field(
        ..., min_length=1,
        description=f"Up to {MAX_BATCH} records, each with the semantics of "
                    "POST /v1/predict.",
        json_schema_extra={"example": [EXAMPLE_RECORD,
                                       {"v012": 27, "v106": "higher",
                                        "v190": "richest", "v025": "urban",
                                        "m14": 8}]})
    fill: Literal["template", "none"] = "template"
    include_anomaly: bool = True
    detail: Literal["compact", "full"] = Field(
        "compact", description="`compact` returns probabilities and flags "
                               "per record; `full` returns the complete "
                               "cascade report for each.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url=f"{ROOT_PATH}/docs")


@app.get("/health", tags=["status"])
def health():
    e = STATE["engine"]
    return {
        "status": "ok" if e else "degraded",
        "api_version": API_VERSION,
        "model_version": e.version if e else None,
        "models_loaded": len(e.targets) if e else 0,
        "uptime_s": int(time.time() - STATE["started"]),
        "auth_required": REQUIRE_KEY,
        "rate_limit_per_minute": RATE_LIMIT,
        "max_batch": MAX_BATCH,
        "error": STATE["error"],
    }


@app.get("/v1/models", tags=["reference"])
def models(e: CascadeEngine = Depends(engine)):
    return {"model_version": e.version, "targets": e.model_summary()}


@app.get("/v1/schema", tags=["reference"])
def schema(e: CascadeEngine = Depends(engine)):
    return e.schema_payload()


@app.get("/v1/presets", tags=["reference"])
def presets():
    return {"presets": [{"id": k, "name": v["name"], "summary": v["summary"],
                         "record": v["record"]} for k, v in PRESETS.items()]}


@app.get("/v1/keys/verify", tags=["predict"])
def verify(info: KeyInfo = Depends(require_key)):
    return {"valid": True, **info.as_dict()}


@app.post("/v1/predict", tags=["predict"])
def predict(body: PredictRequest, info: KeyInfo = Depends(require_key),
            e: CascadeEngine = Depends(engine)):
    try:
        result = e.assess(body.record, fill=body.fill,
                          include_anomaly=body.include_anomaly,
                          include_record=body.include_record)
    except Exception as exc:  # noqa: BLE001 — surface as a clean 400
        log.exception("predict failed")
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail=f"Could not score record: {exc}") from exc
    result["meta"]["key_id"] = info.key_id
    return result


@app.post("/v1/predict/batch", tags=["predict"])
def predict_batch(body: BatchRequest, info: KeyInfo = Depends(require_key),
                  e: CascadeEngine = Depends(engine)):
    if len(body.records) > MAX_BATCH:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"At most {MAX_BATCH} records per request "
                   f"(got {len(body.records)}).")
    t0 = time.perf_counter()
    try:
        if body.detail == "full":
            results = [e.assess(r, fill=body.fill,
                                include_anomaly=body.include_anomaly)
                       for r in body.records]
            report = None
        else:
            results, report = e.assess_many(
                body.records, fill=body.fill,
                include_anomaly=body.include_anomaly)
    except Exception as exc:  # noqa: BLE001
        log.exception("batch predict failed")
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail=f"Could not score batch: {exc}") from exc
    return {
        "n": len(results),
        "results": results,
        "inputs": report,
        "meta": {"model_version": e.version, "key_id": info.key_id,
                 "latency_ms": round((time.perf_counter() - t0) * 1000, 1)},
    }
