"""The shippable detector: models + allowlist + threshold + explanations.

This is the one object the API (stage 3) loads. It wraps everything learned in
notebooks 01 to 05:

  * an ensemble of host models (averaged probabilities)
  * a reputation allowlist: popular, non-platform registered domains are trusted,
    the way real browser filters work
  * a decision threshold chosen on held-out data (notebook 05)
  * human-readable reasons for every verdict
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import SHORTENERS, extract_host_features, split_host


def allowlist_from_tranco(path: Path, top_k: int = 10_000) -> set[str]:
    """Registered domains of the top_k Tranco sites, minus shorteners and platforms.

    Platforms are left out on purpose: trusting weebly.com must never mean
    trusting anything.weebly.com.
    """
    df = pd.read_csv(path, names=["rank", "domain"], header=None, nrows=top_k)
    allow = set()
    for d in df["domain"].astype(str):
        p = split_host(d)
        if not p["platform"] and p["registered"] not in SHORTENERS:
            allow.add(p["registered"])
    return allow


class PhishDetector:
    """Average of several models' P(phishing), with an allowlist and a threshold."""

    def __init__(self, models: dict, allowlist=(), threshold: float = 0.5, weights=None,
                 version: str = "v1"):
        self.models = dict(models)
        self.weights = weights or {name: 1.0 for name in self.models}
        self.allowlist = set(allowlist)
        self.threshold = threshold
        self.version = version
        self.metadata: dict = {}

    # ----- scoring -----------------------------------------------------------
    def fit(self, urls, y):
        for m in self.models.values():
            m.fit(urls, y)
        return self

    def model_scores(self, urls) -> pd.DataFrame:
        """Each component model's P(phishing), one column per model."""
        urls = [str(u) for u in urls]
        return pd.DataFrame({name: m.phish_proba(urls) for name, m in self.models.items()})

    def ensemble_proba(self, urls) -> np.ndarray:
        """Weighted average of the models, ignoring the allowlist."""
        s = self.model_scores(urls)
        w = pd.Series(self.weights)[s.columns]
        return (s * w).sum(axis=1).values / w.sum()

    def is_allowlisted(self, url: str) -> bool:
        p = split_host(url)
        return (not p["platform"]) and p["registered"] in self.allowlist

    def phish_proba(self, urls) -> np.ndarray:
        """Final score: the ensemble, forced to 0 for allowlisted domains."""
        urls = [str(u) for u in urls]
        p = self.ensemble_proba(urls)
        allowed = np.array([self.is_allowlisted(u) for u in urls])
        return np.where(allowed, 0.0, p)

    def predict(self, urls) -> np.ndarray:
        return (self.phish_proba(urls) >= self.threshold).astype(int)

    # ----- explanations ------------------------------------------------------
    def explain(self, url: str) -> dict:
        """Verdict, score, and plain-English reasons for one URL."""
        parts = split_host(url)
        feats = extract_host_features(url)
        per_model = self.model_scores([url]).iloc[0].round(3).to_dict()
        allowed = self.is_allowlisted(url)
        score = 0.0 if allowed else float(self.ensemble_proba([url])[0])

        reasons = []
        if allowed:
            reasons.append(f"{parts['registered']} is on the popular-domain allowlist")
        if parts["platform"]:
            reasons.append(f"hosted on a free platform; the site name '{parts['owner']}' was chosen by whoever signed up")
        if feats["brand_mismatch"]:
            reasons.append("contains a brand name, but the site isn't owned by that brand")
        if feats["is_shortener"]:
            reasons.append("URL shortener; the real destination is hidden")
        if feats["has_ip"]:
            reasons.append("uses a raw IP address instead of a domain name")
        if feats["n_suspicious_words"]:
            reasons.append("hostname contains words common in phishing (login, verify, secure...)")
        if feats["owner_digit_ratio"] > 0.3:
            reasons.append("site name is mostly digits")

        return {
            "url": url,
            "host": parts["host"],
            "registered_domain": parts["registered"],
            "verdict": "phishing" if score >= self.threshold else "legit",
            "score": round(score, 4),
            "threshold": self.threshold,
            "allowlisted": allowed,
            "model_scores": per_model,
            "reasons": reasons,
        }

    # ----- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        """Save the detector (joblib) plus a human-readable JSON card next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        card = {
            "version": self.version,
            "saved": date.today().isoformat(),
            "threshold": self.threshold,
            "models": list(self.models),
            "weights": self.weights,
            "allowlist_size": len(self.allowlist),
            **self.metadata,
        }
        path.with_suffix(".json").write_text(json.dumps(card, indent=2, default=str))

    @staticmethod
    def load(path: Path) -> "PhishDetector":
        return joblib.load(path)
