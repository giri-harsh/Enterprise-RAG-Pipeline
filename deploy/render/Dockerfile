# syntax=docker/dockerfile:1
# ==============================================================================
# Render image — GEMINI path (root-cause memory fix, not a host swap).
#
# Local embeddings (sentence-transformers/torch, then fastembed/onnxruntime as a
# first attempt at trimming that) were the dominant RAM cost — ~1 GB and ~650 MB
# resident respectively, measured. Gemini embeddings replace the loaded model
# with an HTTP call: no embedding model in this process's memory at all.
#
#   embeddings : Gemini API (gemini-embedding-2-preview, 3072-dim) — HTTP, no
#                local model, no torch, no onnxruntime embedding pass
#   reranker   : FlashRank ms-marco-MiniLM-L-12-v2 (audited default — restored,
#                no longer needs trimming once the embedding model is gone)
#   vectors    : embedded Qdrant, 3072-dim collection baked into the image
#   llm        : direct Groq (openai/gpt-oss-120b / -20b)
#
# GEMINI_API_KEY is a runtime secret (Render env var), never baked into the
# image or this file. The image build does NOT call Gemini — .qdrant_local is
# ingested locally beforehand (where the key is available as an env var) and
# copied in already populated, same pattern as the embedded-Qdrant Dockerfiles
# before it.
#
# See DEPLOYMENT_REPORT.md for the before/after memory measurement.
# ==============================================================================
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Kept from the debugging pass that found the real cause below — catches
    # SIGILL/SIGSEGV and prints a Python-level stack frame instead of a bare
    # exit code, cheap insurance for the next time something native crashes.
    PYTHONFAULTHANDLER=1

WORKDIR /app

# requirements-prod.txt already targets this path (it excludes
# sentence-transformers and includes langchain-google-genai — see the header
# comment in that file). No extra packages needed, unlike the fastembed variant.
#
# nemoguardrails is pinned to 0.23.0 here specifically (see the pin's comment
# in requirements-prod.txt): 0.21.0's BasicEmbeddingsIndex used Annoy's native
# C++ extension for the canonical-form/flow search index, and Annoy SIGILLs
# (exit 132) on Render's virtualized CPU — 100% reproducible on every
# guardrail check, confirmed via PYTHONFAULTHANDLER. 0.23.0 replaced Annoy
# with a pure-numpy cosine-similarity index; no native ANN library at all.
# Chased numpy version/CPU-dispatch theories first (OPENBLAS_CORETYPE,
# NPY_DISABLE_CPU_FEATURES, building numpy from source) before the
# faulthandler traceback's exact file path revealed the installed nemoguardrails
# version differed from local — numpy was never the actual problem.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements-prod.txt

# Pre-fetch the FlashRank ONNX reranker model into the image.
RUN python -c "from flashrank import Ranker; Ranker(cache_dir='/tmp/flashrank')"

COPY app/ ./app/
# Populated locally with `python -m app.ingestion.processor DATA --wipe` under
# USE_LOCAL_EMBEDDINGS=false (Gemini, 3072-dim) before building this image.
COPY .qdrant_local/ ./.qdrant_local/

RUN useradd --create-home --shell /bin/false appuser \
    && mkdir -p /tmp/flashrank \
    && chown -R appuser:appuser /app /tmp/flashrank
USER appuser

ENV LOCAL_MODE=false \
    USE_LOCAL_QDRANT=true \
    USE_LOCAL_EMBEDDINGS=false \
    USE_FASTEMBED=false \
    USE_GATEWAY=false \
    GEMINI_EMBED_MODEL=models/gemini-embedding-2-preview \
    QDRANT_LOCAL_PATH=/app/.qdrant_local \
    GROQ_MODEL=openai/gpt-oss-120b \
    GROQ_GUARD_MODEL=openai/gpt-oss-20b \
    LOGFIRE_SEND_TO_LOGFIRE=false \
    LANGSMITH_TRACING=false

EXPOSE 8080
# Render injects $PORT.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
