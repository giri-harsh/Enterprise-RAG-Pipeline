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
    # numpy's OpenBLAS backend auto-detects CPU SIMD features (AVX-512 etc.) at
    # runtime. On Render's shared/virtualized CPUs the reported feature set can
    # advertise instructions the actual host doesn't execute cleanly, which
    # crashes the process with SIGILL (exit 132) the first time a numpy op
    # that dispatches to that codepath runs — observed here inside NeMo
    # Guardrails' embedding-similarity search (a numpy op), not anywhere
    # memory-related. Forcing a conservative baseline target avoids the
    # mis-dispatch. See DEPLOYMENT_REPORT.md.
    OPENBLAS_CORETYPE=Haswell \
    # Diagnostic: OPENBLAS_CORETYPE alone didn't stop the crash — same fault
    # point, same exit code. faulthandler catches SIGILL too (on platforms
    # that support it) and prints the actual Python-level stack frame that
    # was executing, turning a blind native-code guess into a real traceback
    # in the container logs. Temporary — remove once the real cause is found.
    PYTHONFAULTHANDLER=1

WORKDIR /app

# requirements-prod.txt already targets this path (it excludes
# sentence-transformers and includes langchain-google-genai — see the header
# comment in that file). No extra packages needed, unlike the fastembed variant.
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
