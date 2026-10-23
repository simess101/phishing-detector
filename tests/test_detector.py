import zipfile

import numpy as np

from phishdetect.detector import PhishDetector, allowlist_from_tranco
from phishdetect.models import HostCharNgram, host_psl_boosted

URLS = ["https://www.example.com", "https://docs.example.org", "https://www.vcu.edu",
        "https://mail.yahoo.com", "http://paypal-verify.xyz/login", "https://apple-id.webflow.io",
        "http://secure-bank-login.top/update", "https://x7k2.pages.dev/"] * 15
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 15)


def _detector(**kw):
    models = {"E": host_psl_boosted(), "D": HostCharNgram(min_df=1)}
    return PhishDetector(models, **kw).fit(URLS, Y)


def test_scores_in_range_and_ensemble_is_average():
    det = _detector()
    s = det.model_scores(["https://github.com", "http://paypal-login.top"])
    p = det.ensemble_proba(["https://github.com", "http://paypal-login.top"])
    assert np.allclose(p, s.mean(axis=1))
    assert ((p >= 0) & (p <= 1)).all()


def test_allowlist_forces_legit_but_never_for_platform_pages():
    det = _detector(allowlist={"paypal-verify.xyz", "webflow.io"}, threshold=0.5)
    assert det.phish_proba(["http://paypal-verify.xyz/login"])[0] == 0.0
    # webflow.io on the allowlist must not whitelist someone's page on it
    assert not det.is_allowlisted("https://apple-id.webflow.io")


def test_explain_has_reasons():
    det = _detector(threshold=0.5)
    out = det.explain("https://paypal-account.webflow.io/login")
    assert out["verdict"] in {"phishing", "legit"}
    assert set(out["model_scores"]) == {"E", "D"}
    assert any("free platform" in r for r in out["reasons"])
    assert any("brand" in r for r in out["reasons"])


def test_save_and_load_roundtrip(tmp_path):
    det = _detector(threshold=0.42)
    det.metadata["note"] = "test"
    det.save(tmp_path / "det.joblib")
    loaded = PhishDetector.load(tmp_path / "det.joblib")
    assert loaded.threshold == 0.42
    assert (tmp_path / "det.json").exists()
    u = ["http://paypal-verify.xyz/login"]
    assert np.allclose(loaded.phish_proba(u), det.phish_proba(u))


def test_allowlist_from_tranco_skips_platforms_and_shorteners(tmp_path):
    path = tmp_path / "t.csv.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("top-1m.csv", "1,google.com\n2,bit.ly\n3,blogspot.com\n4,bbc.co.uk\n5,far.com")
    allow = allowlist_from_tranco(path, top_k=4)
    assert allow == {"google.com", "blogspot.com", "bbc.co.uk"}
