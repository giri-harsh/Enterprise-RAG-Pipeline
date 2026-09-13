#!/usr/bin/env bash
# ==============================================================================
# One-shot Cloud Run deploy for Enterprise RAG Pipeline.
#
# PREREQUISITES you must supply (see DEPLOYMENT_REPORT.md "Blocked"):
#   1. Google Cloud SDK installed and authenticated:
#        gcloud auth login
#        gcloud auth application-default login
#   2. A GCP project with billing enabled:
#        export GCP_PROJECT=your-project-id
#
# Everything else (mode flags, model overrides, secrets) is wired below.
# ==============================================================================
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT to your billing-enabled project id}"
REGION="${REGION:-us-central1}"
SERVICE="enterprise-rag-api"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

gcloud config set project "$GCP_PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
       secretmanager.googleapis.com artifactregistry.googleapis.com

# --- Secrets ------------------------------------------------------------------
# GROQ_API_KEY comes from .env; API_KEY is generated here if absent.
API_KEY_VALUE="$(grep -E '^API_KEY=' .env | cut -d= -f2- || true)"
if [ -z "${API_KEY_VALUE}" ]; then
  API_KEY_VALUE="$(python -c 'import secrets;print(secrets.token_hex(32))')"
  echo "Generated API_KEY: ${API_KEY_VALUE}"
  echo "  -> add this to the Streamlit UI secrets as API_KEY"
fi

put_secret() {  # name value
  printf '%s' "$2" | gcloud secrets create "$1" --data-file=- 2>/dev/null \
    || printf '%s' "$2" | gcloud secrets versions add "$1" --data-file=-
}
put_secret GROQ_API_KEY "$(grep -E '^GROQ_API_KEY=' .env | cut -d= -f2-)"
put_secret API_KEY      "$API_KEY_VALUE"

# --- Build the embedded-path image via Cloud Build ----------------------------
IMAGE="${REGION}-docker.pkg.dev/${GCP_PROJECT}/cloud-run-source-deploy/${SERVICE}:$(date +%s)"
gcloud builds submit --timeout=1800s \
  --tag "$IMAGE" \
  --gcs-log-dir="gs://${GCP_PROJECT}_cloudbuild/logs" \
  -f deploy/gcp/Dockerfile . \
  || gcloud builds submit --timeout=1800s --tag "$IMAGE" .   # fallback: default Dockerfile

# --- Deploy ------------------------------------------------------------------
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --max-instances 1 \
  --min-instances 0 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "GROQ_MODEL=openai/gpt-oss-120b,GROQ_GUARD_MODEL=openai/gpt-oss-20b,LANGSMITH_TRACING=false,LOGFIRE_SEND_TO_LOGFIRE=false" \
  --set-secrets "GROQ_API_KEY=GROQ_API_KEY:latest,API_KEY=API_KEY:latest"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo
echo "Service URL: $URL"
echo "--- /health ---"
curl -s "$URL/health"; echo
echo "--- smoke query ---"
curl -s -X POST "$URL/query" -H 'Content-Type: application/json' \
     -H "X-API-Key: ${API_KEY_VALUE}" \
     -d '{"q":"How do I autoscale pods in Kubernetes?","thread_id":"smoke-test"}' | head -c 600
echo
