import sqlite3
import zipfile

import numpy as np

from phishdetect.benchmark import load_chrome_history, load_openphish, load_tranco
from phishdetect.models import build_models


def test_tranco_samples_head_and_tail(tmp_path):
    csv = "\n".join(f"{i},site{i}.com" for i in range(1, 20_001))
    path = tmp_path / "tranco.csv.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("top-1m.csv", csv)
    df = load_tranco(n=100, path=path)
    assert len(df) == 100
    assert (df["rank"] <= 10_000).sum() == 50
    assert df["url"].str.startswith("https://").all()
    assert (df["is_phishing"] == 0).all()


def test_openphish_keeps_only_http_urls(tmp_path):
    path = tmp_path / "feed.txt"
    path.write_text("http://a.xyz/login\nhttps://b.top/\n\nnot a url\nhttp://a.xyz/login\n")
    df = load_openphish(path=path)
    assert list(df["url"]) == ["http://a.xyz/login", "https://b.top/"]
    assert (df["is_phishing"] == 1).all()


def test_chrome_history_filters_and_caps(tmp_path):
    db = tmp_path / "History"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE urls (id INTEGER PRIMARY KEY, url TEXT)")
    rows = [f"https://mail.google.com/mail/u/0/#{i}" for i in range(10)] + [
        "https://github.com/a/b", "http://localhost:3000/", "http://192.168.1.1/",
        "chrome://settings", "https://intranet/home", "https://printer.local/",
    ]
    con.executemany("INSERT INTO urls (url) VALUES (?)", [(r,) for r in rows])
    con.commit(); con.close()

    df = load_chrome_history(path=db, max_per_host=3)
    hosts = df["url"].str.extract(r"https?://([^/]+)")[0]
    assert set(hosts) == {"mail.google.com", "github.com"}
    assert (hosts == "mail.google.com").sum() == 3


def test_missing_history_returns_empty(tmp_path):
    assert load_chrome_history(path=tmp_path / "nope").empty


def test_models_fit_and_score():
    urls = ["https://www.example.com", "https://www.vcu.edu", "http://paypal-verify.xyz/login",
            "http://secure-bank.top/update"] * 10
    y = np.array([0, 0, 1, 1] * 10)
    for model in build_models().values():
        model.fit(urls, y)
        p = model.phish_proba(["https://github.com/x", "http://apple-id.top/signin"])
        assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()


def test_empty_history_keeps_int_labels(tmp_path):
    import pandas as pd
    other = pd.DataFrame({"url": ["https://a.com"], "source": ["tranco"], "is_phishing": [0]})
    merged = pd.concat([other, load_chrome_history(path=tmp_path / "nope")])
    assert merged["is_phishing"].dtype.kind == "i"
