"""Dataset loading.

Uses the PhiUSIIL Phishing URL dataset from UCI (id 967, CC BY 4.0):
~235k URLs, 134,850 legitimate and 100,945 phishing. In the raw data
label 1 = legitimate and 0 = phishing; we flip it so phishing is the
positive class, which is what precision/recall should be measured on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "phiusiil.csv"
UCI_ID = 967


def load_phiusiil(refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame with columns ['url', 'is_phishing'].

    Downloads once via ucimlrepo and caches to data/raw/phiusiil.csv.
    """
    if RAW_PATH.exists() and not refresh:
        return pd.read_csv(RAW_PATH)

    from ucimlrepo import fetch_ucirepo  # imported lazily, only needed once

    ds = fetch_ucirepo(id=UCI_ID)
    # .original has every column (IDs, features, target) in one frame,
    # so we don't depend on which role UCI assigned the URL column.
    raw = ds.data.original
    df = pd.DataFrame({
        "url": raw["URL"].astype(str),
        "is_phishing": (raw["label"] == 0).astype(int),
    })
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df
