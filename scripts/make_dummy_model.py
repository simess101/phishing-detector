"""Train a tiny stand-in detector so CI can build and smoke-test the Docker image.

The real model is trained in notebook 05 and is not committed to git.
Usage: python scripts/make_dummy_model.py models/detector_v1.joblib
"""
import sys
from pathlib import Path

import numpy as np

from phishdetect.detector import PhishDetector
from phishdetect.models import HostCharNgram, host_psl_boosted

urls = ["https://www.example.com", "https://docs.example.org", "https://www.vcu.edu",
        "https://mail.yahoo.com", "http://paypal-verify.xyz/login", "https://apple-id.webflow.io",
        "http://secure-bank-login.top/update", "https://x7k2.pages.dev/"] * 15
y = np.array([0, 0, 0, 0, 1, 1, 1, 1] * 15)

det = PhishDetector({"E+U (boosted)": host_psl_boosted(), "D+U": HostCharNgram(min_df=1)},
                    allowlist={"google.com"}, threshold=0.5, version="ci-dummy").fit(urls, y)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "models/detector_v1.joblib")
det.save(out)
print("saved", out)
