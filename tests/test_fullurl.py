import importlib.util
import json
import urllib.error
from pathlib import Path

import pandas as pd

from phishdetect.fullurl import cap_per_domain, normalize_full_url, shallow_stats


def test_normalize_removes_collection_differences():
    assert normalize_full_url("HTTPS://www.Example.com/Path/?q=1#top") == "example.com/path/?q=1"
    assert normalize_full_url("https://example.com/") == "example.com"
    assert normalize_full_url("http://www2.example.org/a") == "example.org/a"
    assert normalize_full_url("example.com/" + "a" * 500) == ("example.com/" + "a" * 500)[:200]
    # identical pages from two sources must look identical
    assert normalize_full_url("https://www.vcu.edu/") == normalize_full_url("http://vcu.edu")


def test_cap_per_domain():
    df = pd.DataFrame({"url": [f"https://a.com/{i}" for i in range(10)] + ["https://b.org/x"]})
    out = cap_per_domain(df, k=3)
    assert (out["url"].str.contains("a.com")).sum() == 3 and len(out) == 4


def test_shallow_stats_columns():
    s = shallow_stats(pd.Series(["a.com", "b.org/x.php?id=1", "c.net/p%20q.html"]))
    assert list(s["has_path"]) == [0, 1, 1]
    assert list(s["has_query"]) == [0, 1, 0]
    assert list(s["ends_php"]) == [0, 1, 0]


def _collector():
    path = Path(__file__).resolve().parents[1] / "scripts" / "collect_commoncrawl.py"
    spec = importlib.util.spec_from_file_location("collect_commoncrawl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_query_domain_keeps_only_ok_html():
    mod = _collector()
    lines = [
        {"url": "https://a.com/page", "status": "200", "mime": "text/html"},
        {"url": "https://a.com/page", "status": "200", "mime": "text/html"},   # duplicate
        {"url": "https://a.com/img.png", "status": "200", "mime": "image/png"},
        {"url": "https://a.com/gone", "status": "404", "mime": "text/html"},
        {"url": "https://docs.a.com/x", "status": "200", "mime": "application/xhtml+xml"},
    ]
    fake = lambda url: "\n".join(json.dumps(l) for l in lines) + "\nnot json"
    assert mod.query_domain("CC-TEST", "a.com", fetch=fake) == ["https://a.com/page", "https://docs.a.com/x"]


def test_query_domain_handles_no_captures():
    mod = _collector()
    def fake(url):
        raise urllib.error.HTTPError(url, 404, "no captures", None, None)
    assert mod.query_domain("CC-TEST", "nothing.example", fetch=fake) == []


def test_query_domain_retries_cut_off_responses(monkeypatch):
    import http.client
    mod = _collector()
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    calls = {"n": 0}
    def flaky(url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http.client.IncompleteRead(b"partial")
        if calls["n"] == 2:
            raise urllib.error.HTTPError(url, 504, "gateway timeout", None, None)
        return json.dumps({"url": "https://a.com/ok", "status": "200", "mime": "text/html"})
    assert mod.query_domain("CC-TEST", "a.com", fetch=flaky) == ["https://a.com/ok"]
    assert calls["n"] == 3


def test_query_domain_gives_up_after_retries(monkeypatch):
    import http.client
    mod = _collector()
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    def always_broken(url):
        raise http.client.IncompleteRead(b"")
    import pytest
    with pytest.raises(mod.IndexUnreachable):
        mod.query_domain("CC-TEST", "a.com", fetch=always_broken)


def test_query_domain_skips_refused_domain(monkeypatch):
    mod = _collector()
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    def bad_request(url):
        raise urllib.error.HTTPError(url, 400, "Bad Request", None, None)
    assert mod.query_domain("CC-TEST", "weird..domain", fetch=bad_request) == []
