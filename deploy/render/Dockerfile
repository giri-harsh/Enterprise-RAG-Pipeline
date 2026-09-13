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
        gcc g++ gfortran libgomp1 libopenblas-dev liblapack-dev \
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
    # faulthandler caught the actual fault (SIGILL, exit 132): every crash was
    # inside numpy's OWN compiled dispatcher — nemoguardrails/embeddings/
    # basic.py's build(), specifically np.linalg.norm() — not OpenBLAS (which
    # is why OPENBLAS_CORETYPE alone did nothing: numpy.linalg._umath_linalg
    # has its own independent runtime CPU-feature dispatch, separate from
    # whatever BLAS is linked underneath). Render's virtualized CPU
    # advertises AVX-512 support that its actual execution doesn't handle
    # correctly, so numpy's dispatcher picks an AVX-512 code path and faults.
    # This disables that family and falls back to a safe baseline.
    NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_KNL AVX512_KNM AVX512_SPR AVX512_GNR" \
    # Kept enabled: confirms whether this is the actual fix from the traceback.
    PYTHONFAULTHANDLER=1

WORKDIR /app

# requirements-prod.txt already targets this path (it excludes
# sentence-transformers and includes langchain-google-genai — see the header
# comment in that file). No extra packages needed, unlike the fastembed variant.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements-prod.txt

# Neither numpy version pinned so far (2.4.4, then 1.26.4) fixed the SIGILL —
# same exact fault line on both, which means it isn't a numpy-version issue,
# it's the prebuilt PyPI wheel's compiled SIMD baseline not matching what
# Render's virtualized CPU actually executes correctly. Rebuilding from
# source targets the real build-time CPU instead of a generic wheel baseline.
# --force-reinstall so this genuinely replaces the wheel just installed above.
RUN pip install --no-cache-dir --no-binary numpy --force-reinstall "numpy==1.26.4"

# Belt-and-suspenders: also route NeMo Guardrails' embeddings-index build/
# search (nemoguardrails/embeddings/basic.py) around the two calls that
# actually crashed (confirmed via faulthandler — np.linalg.norm, called from
# build() and search()) onto plain elementwise arithmetic instead of numpy's
# specialized linalg dispatch. Fails the build loudly if the source text
# doesn't match, rather than silently no-op'ing on a nemoguardrails version
# bump.
RUN python -c "\
import nemoguardrails.embeddings.basic as m, inspect; \
path = inspect.getfile(m); \
src = open(path).read(); \
old1 = 'matrix = np.asarray(self._embeddings, dtype=np.float32)\n        norms = np.linalg.norm(matrix, axis=1, keepdims=True)'; \
new1 = 'matrix = np.asarray(self._embeddings, dtype=np.float64)\n        norms = np.sqrt(np.sum(matrix * matrix, axis=1, keepdims=True))'; \
old2 = 'query = np.asarray(_embedding, dtype=np.float32)\n        query_norm = np.linalg.norm(query)'; \
new2 = 'query = np.asarray(_embedding, dtype=np.float64)\n        query_norm = float(np.sqrt(np.sum(query * query)))'; \
old3 = 'cosine = self._index @ query'; \
new3 = 'cosine = np.sum(self._index * query, axis=1)'; \
assert old1 in src and old2 in src and old3 in src, 'nemoguardrails basic.py source changed — patch no longer applies'; \
src = src.replace(old1, new1).replace(old2, new2).replace(old3, new3); \
open(path, 'w').write(src); \
print('patched', path)"

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
