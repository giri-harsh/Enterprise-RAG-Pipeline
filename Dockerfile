FROM python:3.11-slim-bookworm

# Patch OS-level CVEs, then install the toolchain that native wheels need to build.
# libgomp1 is a runtime requirement of the ONNX runtime that FlashRank uses.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Never write .pyc files, never buffer stdout — an unflushed buffer means logs
# vanish when a container is killed, which is exactly when you need them.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first, so the dependency layer is cached and only rebuilds when
# requirements-prod.txt changes — not on every code edit.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements-prod.txt

# Application code only. .dockerignore keeps DATA/, evals/, ui/, notebooks/ and
# DOCS/ out of the image entirely.
COPY app/ ./app/

# Run unprivileged. A container process that does not need root should not have
# it — if the app is compromised, this is the difference between an attacker
# holding a sandboxed user and holding the container.
RUN useradd --create-home --shell /bin/false appuser \
    && mkdir -p /tmp/flashrank \
    && chown -R appuser:appuser /app /tmp/flashrank
USER appuser

EXPOSE 8080

# Cloud Run injects $PORT and is not contractually bound to 8080. Reading it with
# a default keeps the image portable across Cloud Run, Render, Fly and local runs.
#
# Single worker on purpose: LangGraph's MemorySaver keeps conversation state in
# process memory, so a second worker would be a second, separate memory. See
# DEPLOYMENT.md.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
