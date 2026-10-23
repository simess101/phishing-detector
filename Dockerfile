# Phishing detector API
# Build:  docker build -t phishdetect .
# Run:    docker run --rm -p 8000:8000 phishdetect
# Docs:   http://localhost:8000/docs

# Match the Python version the model was trained with (pickles are version-sensitive).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PHISHDETECT_MODEL=/app/models/detector_v1.joblib \
    TLDEXTRACT_CACHE=/tmp/tldextract

WORKDIR /app

# Dependencies first, so code changes don't reinstall everything.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# The package and the trained model (models/ is git-ignored, so the model
# comes from your machine at build time).
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .
COPY models/detector_v1.joblib models/detector_v1.json models/

# Don't run as root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if b'\"model_loaded\":true' in urllib.request.urlopen('http://localhost:8000/health').read() else 1)"

CMD ["uvicorn", "phishdetect.api:app", "--host", "0.0.0.0", "--port", "8000"]
