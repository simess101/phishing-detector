"""Real-world benchmark data, from outside the training dataset.

Three sources:
  * Tranco top-1M list: real, popular domains (legit).
  * Your own browser history: real URLs with subdomains and paths (legit).
  * OpenPhish public feed: currently active phishing URLs.

Everything is cached in data/raw/, which is git-ignored. OpenPhish snapshots
are saved with the date in the filename, so over time you build up a history
you can use to measure drift.

Safety: phishing URLs are only ever handled as text. Never open them.
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from .data import PROJECT_ROOT
from .features import _is_ip, host_only, registered_domain

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
OPENPHISH_URL = "https://openphish.com/feed.txt"
UMBRELLA_URL = "https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"

COLUMNS = ["url", "source", "is_phishing"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"url": pd.Series(dtype=str), "source": pd.Series(dtype=str),
                         "is_phishing": pd.Series(dtype=int)})


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "phishdetect-research/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"downloaded {url} -> {dest.relative_to(PROJECT_ROOT)}")


def load_tranco(n: int = 5000, path: Path | None = None, random_state: int = 42,
                refresh: bool = False) -> pd.DataFrame:
    """Sample n legit domains: half from the top 10k, half from the long tail.

    Mixing in the long tail matters. The top 10k are almost all famous sites,
    which are also the ones most likely to be in the training data already.
    """
    path = Path(path) if path else RAW_DIR / "tranco_top1m.csv.zip"
    if refresh or not path.exists():
        _download(TRANCO_URL, path)
    df = pd.read_csv(path, names=["rank", "domain"], header=None)

    head_pool = df[df["rank"] <= 10_000]
    tail_pool = df[df["rank"] > 10_000]
    n_head = min(n // 2, len(head_pool))
    n_tail = min(n - n_head, len(tail_pool))
    picked = pd.concat([
        head_pool.sample(n_head, random_state=random_state),
        tail_pool.sample(n_tail, random_state=random_state),
    ])
    return pd.DataFrame({
        "url": "https://" + picked["domain"].astype(str).values,
        "source": "tranco",
        "is_phishing": 0,
        "rank": picked["rank"].values,
    })


def load_openphish(path: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Today's OpenPhish feed (one URL per line), cached as a dated snapshot."""
    path = Path(path) if path else RAW_DIR / f"openphish_{date.today().isoformat()}.txt"
    if refresh or not path.exists():
        _download(OPENPHISH_URL, path)
    urls = pd.Series(path.read_text(errors="ignore").splitlines()).str.strip()
    urls = urls[urls.str.match(r"^https?://", case=False)].drop_duplicates()
    return pd.DataFrame({"url": urls.values, "source": "openphish", "is_phishing": 1})


def default_chrome_history() -> Path:
    """Chrome's history database for the default profile on macOS."""
    return Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "Default" / "History"


def load_chrome_history(path: Path | None = None, n: int = 1000, max_per_host: int = 3,
                        random_state: int = 42) -> pd.DataFrame:
    """Sample up to n URLs from your Chrome history, as legit examples.

    Chrome locks its database while running, so we read a temporary copy.
    max_per_host stops one site (Gmail, say) from making up half the sample.
    Local and internal hosts (localhost, IPs, *.local, no dot) are dropped.

    This is personal data. It stays in memory only; nothing is written to disk.
    """
    path = Path(path) if path else default_chrome_history()
    if not path.exists():
        print(f"No Chrome history at {path}; skipping. Pass path= if you use another profile.")
        return _empty()

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "History"
        shutil.copy2(path, copy)
        con = sqlite3.connect(copy)
        try:
            urls = pd.read_sql_query("SELECT url FROM urls", con)["url"]
        finally:
            con.close()

    urls = urls[urls.str.match(r"^https?://", case=False)].drop_duplicates()
    df = pd.DataFrame({"url": urls})
    df["host"] = df["url"].map(host_only)
    internal = (
        ~df["host"].str.contains(".", regex=False)
        | df["host"].map(_is_ip)
        | df["host"].str.endswith((".local", ".internal", ".lan", ".localhost"))
    )
    df = df[~internal]
    df = df.sample(frac=1, random_state=random_state).groupby("host").head(max_per_host)
    df = df.head(n)
    return pd.DataFrame({"url": df["url"].values, "source": "history", "is_phishing": 0})


def load_umbrella(n: int = 50_000, top_k: int = 100_000, exclude_domains=(),
                  path: Path | None = None, random_state: int = 42,
                  refresh: bool = False) -> pd.DataFrame:
    """Sample n legit *hostnames* from the Cisco Umbrella top-1M list, for training.

    Unlike Tranco, Umbrella ranks by DNS lookups, so it lists full hostnames
    (docs.google.com, mail.yahoo.com), which is exactly the legit-with-subdomain
    data PhiUSIIL is missing.

    Only the top_k ranks are used, because the long tail of a DNS list is full of
    trackers, malware callbacks, and junk. Hosts whose registered domain is in
    exclude_domains are dropped, which is how we keep the benchmark unseen.
    """
    path = Path(path) if path else RAW_DIR / "umbrella_top1m.csv.zip"
    if refresh or not path.exists():
        _download(UMBRELLA_URL, path)
    df = pd.read_csv(path, names=["rank", "host"], header=None)
    df = df[df["rank"] <= top_k].copy()
    df["host"] = df["host"].astype(str).str.lower().str.strip(".")
    junk = (
        ~df["host"].str.contains(".", regex=False)
        | df["host"].map(_is_ip)
        | df["host"].str.endswith((".arpa", ".local", ".internal", ".lan", ".localdomain"))
        | df["host"].str.startswith("_")
    )
    df = df[~junk]
    df["registered"] = df["host"].map(registered_domain)
    df = df[~df["registered"].isin(set(exclude_domains))]
    df = df.sample(min(n, len(df)), random_state=random_state)
    return pd.DataFrame({"url": "https://" + df["host"].values, "source": "umbrella",
                         "is_phishing": 0, "rank": df["rank"].values})


def defang(url: str) -> str:
    """Make a URL unclickable for display: hxxps://evil[.]com/login"""
    return str(url).replace("http", "hxxp", 1).replace(".", "[.]")