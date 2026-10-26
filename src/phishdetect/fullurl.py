"""Full-URL data for stage 2 (deep learning on whole URLs, paths included).

Lesson from notebooks 01 to 04: a model that sees the full URL will learn any
difference in how the two classes were *collected*. So every URL from every
source goes through the same normalize_full_url() before training or scoring,
and notebook 06 audits the result for leaks before any model is trained.

Sources:
  * phishing: Phishing.Database "ACTIVE" list on GitHub (~790k URLs, most with paths)
  * legit:    Common Crawl index, collected by scripts/collect_commoncrawl.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .benchmark import RAW_DIR, _download, load_chrome_history, load_openphish, load_tranco
from .features import split_host

PHISHING_DB_URL = ("https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/"
                   "master/phishing-links-ACTIVE.txt")
COMMONCRAWL_PATH = RAW_DIR / "commoncrawl_urls.csv"
SITEMAP_PATH = RAW_DIR / "sitemap_urls.csv"
MAX_LEN = 200

_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
_WWW = re.compile(r"^www\d*\.", re.I)


def normalize_full_url(url: str, max_len: int = MAX_LEN, drop_query: bool = False) -> str:
    """The one view of a URL that every model in stage 2 sees.

    lowercase, no scheme, no leading www., no #fragment, no trailing slash,
    capped at max_len characters:
      'HTTPS://www.Example.com/Path/?q=1#top' -> 'example.com/path/?q=1'
      'https://example.com/'                 -> 'example.com'
    Each step removes a collection difference found between the sources.

    drop_query=True also removes everything after "?" (notebook 08). Sitemap
    URLs almost never have query strings, so a model can learn "? = phishing".
    """
    u = str(url).strip().lower()
    u = _SCHEME.sub("", u)
    u = _WWW.sub("", u)
    u = u.split("#", 1)[0]
    if drop_query:
        u = u.split("?", 1)[0]
    u = u.rstrip("/")
    return u[:max_len]


def benchmark_domains(random_state: int = 42) -> set[str]:
    """Registered domains in the notebook 03 to 05 benchmark.

    Nothing from these domains may enter stage 2 training data, or the
    benchmark stops measuring generalization.
    """
    parts = [load_tranco(n=5000, random_state=random_state)["url"]]
    hist = load_chrome_history(n=1000, max_per_host=3, random_state=random_state)
    parts.append(hist["url"])
    for snap in sorted(RAW_DIR.glob("openphish_*.txt")):
        parts.append(load_openphish(path=snap)["url"])
    urls = pd.concat(parts, ignore_index=True)
    return {split_host(u)["registered"] for u in urls}


def load_phishing_database(path: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Phishing.Database ACTIVE list: one URL per line. Handle as text only."""
    path = Path(path) if path else RAW_DIR / "phishing_database_active.txt"
    if refresh or not path.exists():
        _download(PHISHING_DB_URL, path)
    urls = pd.Series(path.read_text(errors="ignore").splitlines()).str.strip()
    urls = urls[urls.str.match(r"^https?://", case=False)].drop_duplicates()
    return pd.DataFrame({"url": urls.values, "source": "phishing_db", "is_phishing": 1})


def load_commoncrawl(path: Path | None = None) -> pd.DataFrame:
    """Legit URLs gathered by scripts/collect_commoncrawl.py."""
    path = Path(path) if path else COMMONCRAWL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/collect_commoncrawl.py --domains 2500")
    df = pd.read_csv(path)
    df = df[df["url"].astype(str).str.match(r"^https?://", case=False)].drop_duplicates("url")
    return pd.DataFrame({"url": df["url"].values, "source": "commoncrawl", "is_phishing": 0})


def load_sitemaps(path: Path | None = None) -> pd.DataFrame:
    """Legit URLs gathered by scripts/collect_sitemaps.py."""
    path = Path(path) if path else SITEMAP_PATH
    df = pd.read_csv(path)
    df = df[df["url"].astype(str).str.match(r"^https?://", case=False)].drop_duplicates("url")
    return pd.DataFrame({"url": df["url"].values, "source": "sitemap", "is_phishing": 0})


def load_legit() -> pd.DataFrame:
    """All legit full-URL sources that have been collected so far."""
    parts = []
    if COMMONCRAWL_PATH.exists():
        parts.append(load_commoncrawl())
    if SITEMAP_PATH.exists():
        parts.append(load_sitemaps())
    if not parts:
        raise FileNotFoundError("No legit URLs yet. Run: python scripts/collect_sitemaps.py --domains 3000")
    return pd.concat(parts, ignore_index=True).drop_duplicates("url").reset_index(drop=True)


def cap_per_domain(df: pd.DataFrame, k: int, random_state: int = 42) -> pd.DataFrame:
    """Keep at most k URLs per registered domain, so one campaign or one big
    site can't dominate training."""
    reg = df["url"].map(lambda u: split_host(u)["registered"])
    return (df.assign(registered=reg).sample(frac=1, random_state=random_state)
              .groupby("registered").head(k).drop(columns="registered").reset_index(drop=True))


# Shallow, collection-level properties used by the leak audit in notebook 06.
def shallow_stats(normalized: pd.Series) -> pd.DataFrame:
    s = normalized.astype(str)
    return pd.DataFrame({
        "length": s.str.len(),
        "has_path": s.str.contains("/", regex=False).astype(int),
        "n_slashes": s.str.count("/"),
        "has_query": s.str.contains("?", regex=False).astype(int),
        "has_percent": s.str.contains("%", regex=False).astype(int),
        "ends_html": s.str.contains(r"\.html?(?:$|\?)", regex=True).astype(int),
        "ends_php": s.str.contains(r"\.php(?:$|\?)", regex=True).astype(int),
        "n_digits": s.str.count(r"\d"),
    })


class FullUrlTfidf:
    """TF-IDF over 2 to 5 character chunks of the normalized full URL, then
    logistic regression. Same phish_proba(urls) interface as the other models,
    so it can go inside a PhishDetector."""

    def __init__(self, drop_query: bool = False, ngram_range=(2, 5), min_df=3,
                 max_features=300_000, C=4.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.drop_query = drop_query
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=ngram_range, min_df=min_df,
                            sublinear_tf=True, max_features=max_features),
            LogisticRegression(max_iter=2000, C=C),
        )

    def _texts(self, urls):
        return [normalize_full_url(u, drop_query=self.drop_query) for u in urls]

    def fit(self, urls, y):
        self.model.fit(self._texts(urls), y)
        return self

    def phish_proba(self, urls):
        return self.model.predict_proba(self._texts(urls))[:, 1]
