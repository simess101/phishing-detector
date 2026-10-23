"""API tests. They train a tiny detector on the fly, so they need no data files
and run the same on a laptop and in GitHub Actions."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from phishdetect.detector import PhishDetector
from phishdetect.models import HostCharNgram, host_psl_boosted

URLS = ["https://www.example.com", "https://docs.example.org", "https://www.vcu.edu",
        "https://mail.yahoo.com", "http://paypal-verify.xyz/login", "https://apple-id.webflow.io",
        "http://secure-bank-login.top/update", "https://x7k2.pages.dev/"] * 15
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 15)


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    det = PhishDetector({"E": host_psl_boosted(), "D": HostCharNgram(min_df=1)},
                        allowlist={"google.com"}, threshold=0.5, version="test").fit(URLS, Y)
    path = tmp_path_factory.mktemp("model") / "detector.joblib"
    det.save(path)
    monkeypatch_module.setenv("PHISHDETECT_MODEL", str(path))
    from phishdetect.api import app
    with TestClient(app) as c:  # "with" runs the startup that loads the model
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_card(client):
    r = client.get("/model")
    assert r.status_code == 200 and r.json()["version"] == "test"


def test_check_returns_verdict_and_reasons(client):
    r = client.post("/check", json={"url": "https://paypal-account.webflow.io/login"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] in {"phishing", "legit"}
    assert 0 <= body["score"] <= 1
    assert body["model_version"] == "test"
    assert any("free platform" in reason for reason in body["reasons"])


def test_allowlisted_domain_is_legit(client):
    body = client.post("/check", json={"url": "https://docs.google.com/document/d/1"}).json()
    assert body["allowlisted"] and body["verdict"] == "legit" and body["score"] == 0.0


def test_batch(client):
    r = client.post("/check/batch", json={"urls": ["https://www.vcu.edu", "http://paypal-login.top"]})
    assert r.status_code == 200 and len(r.json()) == 2


@pytest.mark.parametrize("payload", [
    {"url": ""},
    {"url": "x" * 3000},
    {"urls": []},
    {"urls": ["https://a.com"] * 101},
    {},
])
def test_bad_input_is_rejected(client, payload):
    endpoint = "/check/batch" if "urls" in payload else "/check"
    assert client.post(endpoint, json=payload).status_code == 422


def test_missing_model_reports_degraded(tmp_path, monkeypatch):
    from phishdetect import api
    monkeypatch.setitem(api._state, "detector", None)
    monkeypatch.setitem(api._state, "error", "FileNotFoundError: nope")
    c = TestClient(api.app)  # no "with": skips startup, keeps the patched state
    assert c.get("/health").json()["status"] == "degraded"
    assert c.post("/check", json={"url": "https://a.com"}).status_code == 503
