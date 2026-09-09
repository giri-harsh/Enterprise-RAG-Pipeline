# syntax=docker/dockerfile:1
# ==============================================================================
# Cloud Run image — EMBEDDED path (embedded Qdrant + local sentence-transformers
# embeddings + direct Groq). Chosen because it needs zero new accounts beyond the
# Groq key that already works. See DEPLOYMENT_REPORT.md for the rationale.
#
# Differences from the repo-root Dockerfile (which targets the Gemini path):
#   - installs sentence-transformers (torch) for local 768-dim embeddings
#   - pre-downloads the all-mpnet-base-v2 weights at build time so the first
#     request does not pay the ~420 MB download inside the Cloud Run timeout
#   - bakes the populated .qdrant_local collection into the image (298 KB)
# ==============================================================================
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf

WORKDIR /app

COPY requirements-prod.txt .
# --prefer-binary keeps torch on its prebuilt wheel. CPU-only torch is enough:
# the reranker and embedder both run on CPU/ONNX.
RUN pip install --no-cache-dir --prefer-binary \
        -r requirements-prod.txt \
        "sentence-transformers==3.3.1" \
        "torch==2.5.1" --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-fetch the embedding model into the image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

COPY app/ ./app/
COPY .qdrant_local/ ./.qdrant_local/

RUN useradd --create-home --shell /bin/false appuser \
    && mkdir -p /tmp/flashrank \
    && chown -R appuser:appuser /app /tmp/flashrank
USER appuser

ENV LOCAL_MODE=true \
    USE_LOCAL_QDRANT=true \
    USE_LOCAL_EMBEDDINGS=true \
    USE_GATEWAY=false \
    QDRANT_LOCAL_PATH=/app/.qdrant_local \
    LOGFIRE_SEND_TO_LOGFIRE=false \
    LANGSMITH_TRACING=false

EXPOSE 8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
