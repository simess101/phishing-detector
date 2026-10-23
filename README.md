# Phishing URL Detector

A machine learning system that flags phishing URLs, built in stages from a scikit-learn baseline up to a deployed API with a browser extension.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Baseline, leakage analysis, real-world benchmark, v1 detector (notebooks 01 to 05) | done |
| 2 | Deep learning: char-level PyTorch model, fine-tuned transformer on emails | planned |
| 3 | Service: FastAPI + Docker + GitHub Actions tests | done |
| 4 | Deploy: AWS, then Kubernetes | planned |
| 5 | Demo: Chrome extension + monitoring dashboard | planned |

## Data

[PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (UCI, CC BY 4.0), about 235k labeled URLs. It downloads automatically on first run and is cached to `data/raw/` (not committed). Phishing is treated as the positive class for all metrics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                           # feature tests should pass
jupyter lab notebooks/01_baseline.ipynb
```

## Layout

```
src/phishdetect/features.py   URL -> numeric features (shared by training, API, extension)
src/phishdetect/data.py       dataset download + caching
notebooks/01_baseline.ipynb   exploration, training, evaluation
tests/                        pytest unit tests
models/                       saved models (not committed)
```

## Results

| model | precision | recall | PR-AUC | notes |
|---|---|---|---|---|
| logreg | | | | |
| random forest | | | | |

## API

The v1 detector is served by FastAPI. It only analyzes the URL as text and never visits it.

```bash
# run locally (after notebook 05 has saved models/detector_v1.joblib)
python -m uvicorn phishdetect.api:app --reload
# or in Docker
docker build -t phishdetect .
docker run --rm -p 8000:8000 phishdetect
```

Interactive docs at http://localhost:8000/docs.

```bash
curl -X POST localhost:8000/check -H "Content-Type: application/json" \
  -d '{"url": "https://secure-login-appleid.webflow.io/"}'
```

```json
{
  "verdict": "phishing",
  "score": 0.98,
  "allowlisted": false,
  "reasons": [
    "hosted on a free platform; the site name 'secure-login-appleid' was chosen by whoever signed up",
    "contains a brand name, but the site isn't owned by that brand",
    "hostname contains words common in phishing (login, verify, secure...)"
  ]
}
```

| endpoint | what it does |
|---|---|
| `GET /health` | service status and whether the model loaded |
| `GET /model` | model card: version, threshold, evaluation numbers |
| `POST /check` | one URL in, verdict + score + reasons out |
| `POST /check/batch` | up to 100 URLs |

## v1 performance and known weaknesses

Measured on a held-out half of a real-world benchmark (my browsing history, Tranco domains, live OpenPhish URLs), with the threshold picked on the other half:

| | v1 |
|---|---|
| false alarms on my browsing history | 1.2% |
| false alarms on long-tail (rank > 10k) sites | 22.4% |
| phishing recall (OpenPhish) | 87.9% |

Known weaknesses: it judges hostnames only, so it can't catch phishing on compromised legitimate sites; it flags most pages on free hosting platforms (github.io, webflow.io) because nearly all such pages in training were phishing; and it is much less accurate on unpopular sites than popular ones. See `JOURNAL.md` for how these numbers were reached.
