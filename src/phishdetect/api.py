"""HTTP API for the phishing detector.

Run locally:
    uvicorn phishdetect.api:app --reload

Endpoints:
    GET  /health        is the service up, and is a model loaded?
    GET  /model         the model card (version, threshold, evaluation numbers)
    POST /check         {"url": "..."}            -> verdict, score, reasons
    POST /check/batch   {"urls": ["...", "..."]}  -> a list of the above (max 100)

The model file is chosen with the PHISHDETECT_MODEL environment variable
(default: models/detector_v1.joblib in the project).

Only the URL string is analyzed. The service never visits the URL.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .detector import PhishDetector

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "detector_v1.joblib"
MAX_URL_LEN = 2048
MAX_BATCH = 100

_state: dict = {"detector": None, "card": {}, "error": None}


def _load(path: Path) -> None:
    try:
        _state["detector"] = PhishDetector.load(path)
        card = path.with_suffix(".json")
        _state["card"] = json.loads(card.read_text()) if card.exists() else {"version": _state["detector"].version}
        _state["error"] = None
    except Exception as e:  # keep the service up so /health can report the problem
        _state["detector"], _state["card"], _state["error"] = None, {}, f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load(Path(os.environ.get("PHISHDETECT_MODEL", DEFAULT_MODEL)))
    yield


app = FastAPI(
    title="Phishing URL Detector",
    description="Scores a URL's hostname for phishing. The URL is analyzed as text and never visited.",
    version="1.0.0",
    lifespan=lifespan,
)


def _validate_url(u: str) -> str:
    u = u.strip()
    if not u:
        raise ValueError("URL is empty")
    if len(u) > MAX_URL_LEN:
        raise ValueError(f"URL is longer than {MAX_URL_LEN} characters")
    return u


class CheckRequest(BaseModel):
    url: str = Field(..., examples=["http://paypal-account-verify.xyz/login"])

    @field_validator("url")
    @classmethod
    def _check(cls, v):
        return _validate_url(v)


class BatchRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)

    @field_validator("urls")
    @classmethod
    def _check(cls, v):
        return [_validate_url(u) for u in v]


class CheckResponse(BaseModel):
    url: str
    host: str
    registered_domain: str
    verdict: str
    score: float
    threshold: float
    allowlisted: bool
    model_scores: dict[str, float]
    reasons: list[str]
    model_version: str
    elapsed_ms: float


def _detector() -> PhishDetector:
    if _state["detector"] is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_state['error']}")
    return _state["detector"]


def _check_one(det: PhishDetector, url: str) -> CheckResponse:
    t0 = time.perf_counter()
    out = det.explain(url)
    return CheckResponse(**out, model_version=det.version,
                         elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.get("/health")
def health():
    ok = _state["detector"] is not None
    return {"status": "ok" if ok else "degraded", "model_loaded": ok, "error": _state["error"]}


@app.get("/model")
def model_card():
    _detector()
    return _state["card"]


@app.post("/check", response_model=CheckResponse)
def check(req: CheckRequest):
    return _check_one(_detector(), req.url)


@app.post("/check/batch", response_model=list[CheckResponse])
def check_batch(req: BatchRequest):
    det = _detector()
    return [_check_one(det, u) for u in req.urls]
