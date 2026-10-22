# Phishing URL Detector

A machine learning system that flags phishing URLs, built in stages from a scikit-learn baseline up to a deployed API with a browser extension.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Baseline: lexical URL features + logistic regression / random forest | in progress |
| 2 | Deep learning: char-level PyTorch model, fine-tuned transformer on emails | planned |
| 3 | Service: FastAPI + Docker + GitHub Actions tests | planned |
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
