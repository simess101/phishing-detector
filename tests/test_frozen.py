import numpy as np
import pandas as pd
import pytest

from phishdetect import frozen
from phishdetect.fullurl import FullUrlTfidf, normalize_full_url


def test_drop_query():
    u = "https://www.Example.com/search/?q=phish&x=1#top"
    assert normalize_full_url(u) == "example.com/search/?q=phish&x=1"
    assert normalize_full_url(u, drop_query=True) == "example.com/search"
    assert normalize_full_url("https://a.com/?x=1", drop_query=True) == "a.com"


def test_full_url_tfidf_ignores_query_when_asked():
    urls = ["https://a.com/blog/post", "https://b.org/docs/page", "http://evil.top/login.php",
            "http://bad.xyz/verify/account"] * 10
    y = np.tile([0, 0, 1, 1], 10)
    m = FullUrlTfidf(drop_query=True, min_df=1).fit(urls, y)
    p = m.phish_proba(["https://a.com/blog/post?id=123", "https://a.com/blog/post"])
    assert abs(p[0] - p[1]) < 1e-9


def _fake_sources(monkeypatch):
    monkeypatch.setattr(frozen, "load_tranco", lambda **k: pd.DataFrame(
        {"url": ["https://a.com", "https://b.org"], "source": "tranco", "is_phishing": 0, "rank": [5, 50_000]}))
    monkeypatch.setattr(frozen, "load_chrome_history", lambda **k: pd.DataFrame(
        {"url": ["https://docs.google.com/x?id=1"], "source": "history", "is_phishing": 0}))
    monkeypatch.setattr(frozen, "load_openphish", lambda **k: pd.DataFrame(
        {"url": ["https://x.webflow.io/login"], "source": "openphish", "is_phishing": 1}))


def test_freeze_and_load_roundtrip(tmp_path, monkeypatch):
    _fake_sources(monkeypatch)
    path = tmp_path / "bench.csv.gz"
    built = frozen.freeze_benchmark(path=path, openphish_path=tmp_path / "op.txt")
    loaded = frozen.load_benchmark(path)
    assert list(loaded["url"]) == list(built["url"])
    assert loaded["is_phishing"].tolist() == [0, 0, 0, 1]
    assert loaded["platform"].tolist() == [False, False, False, True]
    assert np.isnan(loaded.loc[2, "rank"]) and loaded.loc[1, "rank"] == 50_000
    assert set(loaded["split"]) <= {"calib", "eval"}
    assert (tmp_path / "bench.json").exists()


def test_freeze_refuses_to_overwrite(tmp_path, monkeypatch):
    _fake_sources(monkeypatch)
    path = tmp_path / "bench.csv.gz"
    frozen.freeze_benchmark(path=path, openphish_path=tmp_path / "op.txt")
    with pytest.raises(FileExistsError):
        frozen.freeze_benchmark(path=path, openphish_path=tmp_path / "op.txt")
