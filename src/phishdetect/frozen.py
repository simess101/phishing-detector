"""A frozen copy of the real-world benchmark.

Until notebook 07, every notebook rebuilt the benchmark live: Chrome history kept
growing and OpenPhish was re-downloaded, so the same model scored differently on
different days (v1 went from 1.2% to 5.8% history false alarms without changing).

freeze_benchmark() builds it once and saves it to data/processed/ (git-ignored,
since it contains browsing history and live phishing URLs). Every notebook from 08
on calls load_benchmark(), so all scores are measured on exactly the same URLs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from .benchmark import RAW_DIR, load_chrome_history, load_openphish, load_tranco
from .data import PROJECT_ROOT
from .features import split_host

FROZEN_PATH = PROJECT_ROOT / "data" / "processed" / "benchmark_frozen.csv.gz"


def eval_half(domain: str) -> str:
    """Same domain-hash split as notebooks 05 and 07: calib or eval."""
    return "calib" if int(hashlib.md5(domain.encode()).hexdigest(), 16) % 2 == 0 else "eval"


def freeze_benchmark(path: Path = FROZEN_PATH, openphish_path: Path | None = None,
                     random_state: int = 42, overwrite: bool = False) -> pd.DataFrame:
    """Build the benchmark once and save it. Refuses to overwrite unless asked,
    because changing it silently would break comparisons with earlier notebooks."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use load_benchmark(), or overwrite=True on purpose.")

    if openphish_path is None:
        snaps = sorted(RAW_DIR.glob("openphish_*.txt"))
        if not snaps:
            raise FileNotFoundError("No OpenPhish snapshot in data/raw/. Run notebook 03 first.")
        openphish_path = snaps[-1]

    bench = pd.concat([
        load_tranco(n=5000, random_state=random_state),
        load_chrome_history(n=1000, max_per_host=3, random_state=random_state),
        load_openphish(path=openphish_path),
    ], ignore_index=True)
    bench["is_phishing"] = bench["is_phishing"].astype(int)
    parts = pd.DataFrame([split_host(u) for u in bench["url"]])
    bench["registered"] = parts["registered"].values
    bench["platform"] = parts["platform"].values
    bench["split"] = bench["registered"].map(eval_half)

    path.parent.mkdir(parents=True, exist_ok=True)
    bench.to_csv(path, index=False)
    card = {
        "frozen_on": date.today().isoformat(),
        "openphish_snapshot": Path(openphish_path).name,
        "counts": bench.groupby(["source", "split"]).size().unstack().fillna(0).astype(int).to_dict(),
    }
    path.with_suffix("").with_suffix(".json").write_text(json.dumps(card, indent=2))
    return bench


def load_benchmark(path: Path = FROZEN_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run freeze_benchmark() once (notebook 08 does this).")
    df = pd.read_csv(path, keep_default_na=False)     # keep "" as text, not NaN (URLs)
    df["rank"] = pd.to_numeric(df["rank"].astype(object), errors="coerce").astype(float)
    df["is_phishing"] = df["is_phishing"].astype(int)
    df["platform"] = df["platform"].astype(str).str.lower().eq("true")
    return df
