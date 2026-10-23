"""Lexical URL features.

Everything here is computed from the URL string alone, with no network calls,
so the same function works for training, the API, and the browser extension.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from functools import lru_cache
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


# ---------------------------------------------------------------------------
# Host features v2: parse the hostname with the Public Suffix List
# ---------------------------------------------------------------------------
#
# The PSL knows where the "registered" part of a hostname starts:
#   docs.google.com          -> sub "docs",      domain "google",          suffix "com"
#   bbc.co.uk                -> sub "",          domain "bbc",             suffix "co.uk"
#   gemini-login-b.webflow.io -> sub "",         domain "gemini-login-b",  suffix "webflow.io"
# Its "private" section lists shared hosting (webflow.io, workers.dev, web.app,
# github.io), where the label in front belongs to whoever signed up, not to the
# platform. Some platforms phishers use aren't in the PSL, so EXTRA_PLATFORMS
# covers those.

BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "amazon",
    "google", "gmail", "netflix", "facebook", "instagram", "whatsapp", "telegram",
    "linkedin", "chase", "wellsfargo", "bankofamerica", "coinbase", "binance",
    "metamask", "roblox", "steam", "fedex", "usps", "docusign", "adobe", "dropbox",
    "verizon", "gemini",
)
# Short names like "ups", "att", "irs", "live" are left out on purpose: they match
# inside ordinary words ("groups", "delivery", "first") and would add noise.
EXTRA_PLATFORMS = {
    "gitbook.io", "4everland.app", "craftum.io", "weebly.com", "wixsite.com", "square.site",
    "godaddysites.com", "glitch.me", "framer.website", "framer.app", "webwave.dev",
    "mybluehost.me", "teachable.com", "notion.site", "sites.google.com", "jimdosite.com",
    "strikingly.com", "carrd.co", "ipfs.io", "dweb.link", "w3s.link",
    # added after notebook 04's error analysis
    "teleporthq.app", "edgeone.dev", "edgeone.app", "ngrok-free.app", "ngrok.app",
    "ngrok.io", "plesk.page", "g.page",
}

_TLD_EXTRACT = None


def _extractor():
    global _TLD_EXTRACT
    if _TLD_EXTRACT is None:
        import tldextract
        # suffix_list_urls=() uses the PSL snapshot bundled with tldextract:
        # no network call, and the same answer on every machine.
        _TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
    return _TLD_EXTRACT


def split_host(url: str) -> dict:
    """Split a URL's hostname into its parts.

    Returns sub, domain, suffix, registered (domain.suffix), platform (True when
    the site lives on shared hosting), and owner: the label the site's owner
    actually chose. On shared hosting that's the label in front of the platform,
    which is what a phisher picks.
    """
    return dict(_split_cached(host_only(url)))


@lru_cache(maxsize=500_000)
def _split_cached(host: str) -> tuple:
    r = _extractor()(host)
    sub, domain, suffix = r.subdomain, r.domain, r.suffix
    if not domain and suffix:
        # The host IS a platform's own domain (blogspot.com, cloudflare-ipfs.com):
        # treat it as an ordinary site owned by the platform, not as a user page.
        domain, suffix = suffix.split(".", 1) if "." in suffix else (suffix, "")
        registered, platform = r.suffix, False
    else:
        registered = f"{domain}.{suffix}" if domain and suffix else (domain or suffix)
        platform = bool(getattr(r, "is_private", False))
    owner = domain
    if registered in EXTRA_PLATFORMS and sub:
        # A user page on a platform the PSL doesn't list (abc.gitbook.io).
        owner = sub.split(".")[-1]
        sub = ".".join(sub.split(".")[:-1])
        registered = f"{owner}.{registered}"
        platform = True
    return tuple({"host": host, "sub": sub, "domain": domain, "suffix": suffix,
                  "registered": registered, "platform": platform, "owner": owner}.items())


def registered_domain(url: str) -> str:
    return split_host(url)["registered"]


HOST_FEATURE_NAMES = [
    "host_len", "n_sub_labels", "sub_len", "sub_entropy", "sub_digits",
    "owner_len", "owner_entropy", "owner_digits", "owner_digit_ratio", "owner_hyphens",
    "suffix_len", "suffix_labels", "is_cctld", "unknown_suffix",
    "is_free_hosting", "has_ip", "is_shortener",
    "brand_in_host", "brand_mismatch", "n_suspicious_words",
]


def extract_host_features(url: str) -> dict:
    """Numeric features from the hostname only, built on the PSL split."""
    p = split_host(url)
    host, sub, owner, suffix = p["host"], p["sub"], p["owner"], p["suffix"]
    sub_labels = [s for s in sub.split(".") if s]
    owner_digits = sum(ch.isdigit() for ch in owner)
    brand_hits = [b for b in BRANDS if b in host]
    # A brand appears somewhere in the host, but the owner label isn't that brand:
    # paypal-verify.xyz, paypal.com.evil.ru, apple-id.webflow.io.
    brand_mismatch = any(b != owner for b in brand_hits)
    return {
        "host_len": len(host),
        "n_sub_labels": len(sub_labels),
        "sub_len": len(sub),
        "sub_entropy": _entropy(sub),
        "sub_digits": sum(ch.isdigit() for ch in sub),
        "owner_len": len(owner),
        "owner_entropy": _entropy(owner),
        "owner_digits": owner_digits,
        "owner_digit_ratio": owner_digits / len(owner) if owner else 0.0,
        "owner_hyphens": owner.count("-"),
        "suffix_len": len(suffix),
        "suffix_labels": len([s for s in suffix.split(".") if s]),
        "is_cctld": int(len(suffix.rsplit(".", 1)[-1]) == 2),
        "unknown_suffix": int(not suffix),
        "is_free_hosting": int(p["platform"]),
        "has_ip": int(_is_ip(host)),
        "is_shortener": int(p["registered"] in SHORTENERS),
        "brand_in_host": int(bool(brand_hits)),
        "brand_mismatch": int(brand_mismatch),
        "n_suspicious_words": sum(w in host for w in SUSPICIOUS_WORDS),
    }


def featurize_host(urls) -> pd.DataFrame:
    return pd.DataFrame([extract_host_features(str(u)) for u in urls], columns=HOST_FEATURE_NAMES)