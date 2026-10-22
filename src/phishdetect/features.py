"""Lexical URL features.

Everything here is computed from the URL string alone, with no network calls,
so the same function works for training, the API, and the browser extension.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import pandas as pd

SUSPICIOUS_WORDS = (
    "login", "signin", "verify", "account", "update", "secure", "bank",
    "confirm", "password", "wallet", "billing", "support", "webscr",
)
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rebrand.ly", "shorturl.at",
}

FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len",
    "n_dots", "n_hyphens", "n_digits", "n_special", "n_subdomains",
    "digit_ratio", "entropy",
    "has_https", "has_ip", "has_at", "has_port", "has_double_slash_path",
    "is_shortener", "n_suspicious_words", "tld_len",
]


def _parse(url: str):
    url = url.strip()
    # urlsplit needs a scheme to find the host
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return urlsplit(url)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def extract_features(url: str) -> dict:
    """Return a dict of numeric features for one URL."""
    parts = _parse(url)
    host = (parts.hostname or "").lower()
    path, query = parts.path or "", parts.query or ""
    lower = url.lower()
    labels = [p for p in host.split(".") if p]

    try:
        has_port = parts.port is not None
    except ValueError:  # malformed port like :abc
        has_port = True

    n_digits = sum(ch.isdigit() for ch in url)
    return {
        "url_len": len(url),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "n_dots": url.count("."),
        "n_hyphens": url.count("-"),
        "n_digits": n_digits,
        "n_special": sum(url.count(c) for c in "@?=&%~_!*,;$+"),
        "n_subdomains": max(len(labels) - 2, 0),
        "digit_ratio": n_digits / len(url) if url else 0.0,
        "entropy": _entropy(url),
        "has_https": int(parts.scheme == "https"),
        "has_ip": int(_is_ip(host)),
        "has_at": int("@" in url),
        "has_port": int(has_port),
        "has_double_slash_path": int("//" in path),
        "is_shortener": int(host in SHORTENERS),
        "n_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "tld_len": len(labels[-1]) if labels else 0,
    }


def featurize(urls) -> pd.DataFrame:
    """Vectorized helper: list/Series of URLs -> feature DataFrame."""
    return pd.DataFrame([extract_features(str(u)) for u in urls], columns=FEATURE_NAMES)


_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_WWW_RE = re.compile(r"^www\d*\.", re.IGNORECASE)


def normalize_url(url: str) -> str:
    """Strip the scheme and a leading 'www.' so neither can leak the label.

    'https://www.PayPal.com/login' -> 'PayPal.com/login'
    """
    url = _SCHEME_RE.sub("", str(url).strip())
    return _WWW_RE.sub("", url)


def host_only(url: str) -> str:
    """Just the hostname, lowercased, without 'www.'.

    'https://www.paypal.com/login?x=1' -> 'paypal.com'
    """
    host = (_parse(str(url)).hostname or "").lower()
    return _WWW_RE.sub("", host)
