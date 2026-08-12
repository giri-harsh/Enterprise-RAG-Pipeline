# Deployment

Backend on Cloud Run, UI on Streamlit Community Cloud, vectors in Qdrant Cloud.
Everything below fits inside free tiers.

```
Streamlit Community Cloud  ──HTTPS──▶  Cloud Run (FastAPI)  ──▶  Qdrant Cloud
        ui/app.py                          app/main.py              (vectors)
                                                │
                                                ├──▶ Portkey ──▶ Groq
                                                └──▶ Gemini (embeddings)
```

---

## The one architectural constraint

**The backend must run as a single instance.**

Conversation memory lives in LangGraph's `MemorySaver`, which keeps checkpoints in
the process's own RAM keyed by `thread_id`. That is what lets the app work with no
database at all. The consequence is that memory is per-process:

- A restart loses every conversation.
- Two instances mean two separate memories. A user's follow-up question routed to
  the other instance arrives with no history, and the agent answers as if the
  conversation never happened.

So `--max-instances=1` below is deliberate, not an oversight. It caps throughput
at whatever one container handles, which for a portfolio project is irrelevant and
for a real product would not be.

**If you needed to remove that limit,** swap the checkpointer in
`app/agents/graph.py` for a persistent one — `PostgresSaver` or `RedisSaver` share
the same interface, so it is a one-line change plus a managed database. That is the
answer to give when an interviewer asks how you would scale this, and it is a much
better answer than not having noticed the problem.

---

## Prerequisites

| Service | Sign-up | Free tier |
|---|---|---|
| Qdrant Cloud | cloud.qdrant.io | 1 GB cluster |
| Groq | console.groq.com | rate-limited, no card |
| Portkey | app.portkey.ai | 10k requests/month |
| Google AI Studio | aistudio.google.com/apikey | Gemini embeddings |
| Logfire | logfire.pydantic.dev | optional |
| Google Cloud | console.cloud.google.com | Cloud Run scales to zero |

Portkey needs two virtual keys created in its dashboard — `rag` (primary) and
`brag` (fallback) — each holding a *different* Groq API key. Sharing one key
across both makes the fallback useless, since a rate limit would hit both targets
at once. Override the slug names with `PORTKEY_PRIMARY_SLUG` / `PORTKEY_FALLBACK_SLUG`.

---

## 1. Index the corpus

Run from your machine, not the container. Ingestion is a one-off batch job and
there is no reason to pay for a container to sit idle around it.

```bash
cp .env.example .env      # fill in your keys
pip install -r requirements.txt
python -m app.ingestion.processor DATA --wipe
```

`--wipe` drops and recreates the collection, which you need whenever the embedding
model changes — vector width is fixed at collection creation and Qdrant rejects a
query whose dimensions do not match.

Confirm it worked before going further:

```bash
python -c "
from app.services.retrieval.qdrant_service import get_client
from app.config import settings
print(get_client().get_collection(settings.QDRANT_COLLECTION))
"
```

Check `vectors_count` is non-zero and `size` is 3072 (Gemini). If it reads 768 the
Gemini probe failed and you indexed with the local fallback — fix `GEMINI_API_KEY`
and re-run with `--wipe`.

## 2. Verify locally

```bash
uvicorn app.main:app --reload --port 8000     # terminal 1
streamlit run ui/app.py                        # terminal 2
pytest                                         # terminal 3
```

`curl localhost:8000/health` should report `"status": "ok"`. If it says
`"guardrails": "not initialised"`, startup failed — check the logs, and do not
deploy, because the gate is what stops the endpoint being an open LLM proxy.

## 3. Deploy the backend

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
```

Store secrets in Secret Manager rather than passing them as plain environment
variables — env vars are visible to anyone with console read access on the service.

```bash
for KEY in GROQ_API_KEY PORTKEY_API_KEY QDRANT_API_KEY GEMINI_API_KEY LOGFIRE_TOKEN API_KEY; do
  printf "%s" "$(grep "^$KEY=" .env | cut -d= -f2-)" \
    | gcloud secrets create "$KEY" --data-file=- 2>/dev/null \
    || printf "%s" "$(grep "^$KEY=" .env | cut -d= -f2-)" \
       | gcloud secrets versions add "$KEY" --data-file=-
done
```

Generate a real `API_KEY` first if you have not — `openssl rand -hex 32`.

```bash
gcloud run deploy enterprise-rag-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --max-instances 1 \
  --min-instances 0 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars "QDRANT_CLUSTER_ENDPOINT=https://your-cluster.cloud.qdrant.io:6333,LANGSMITH_TRACING=false" \
  --set-secrets "GROQ_API_KEY=GROQ_API_KEY:latest,PORTKEY_API_KEY=PORTKEY_API_KEY:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,LOGFIRE_TOKEN=LOGFIRE_TOKEN:latest,API_KEY=API_KEY:latest"
```

Why these numbers:

- **`--max-instances 1`** — the memory constraint above. Also caps spend.
- **`--memory 2Gi`** — FlashRank loads an ONNX model into memory; 1 GiB is tight
  and shows up as OOM kills on the first reranked query rather than at startup.
- **`--timeout 300`** — a cold start pays for the NeMo rails compile plus the
  FlashRank model download. The default 60s can cut that off.
- **`--allow-unauthenticated`** — the Streamlit UI cannot present a Google
  identity token, so IAM auth is not an option. `API_KEY` is what actually
  protects the endpoint; without it the URL is a free LLM for anyone who finds it.

Verify:

```bash
URL=$(gcloud run services describe enterprise-rag-api --region us-central1 --format='value(status.url)')
curl "$URL/health"
curl -X POST "$URL/query" -H "Content-Type: application/json" -H "X-API-Key: YOUR_KEY" \
     -d '{"q":"How do I autoscale pods?","thread_id":"smoke-test"}'
```

`/health` must report `"auth": "enabled"`. If it says `disabled`, the secret did
not bind and the endpoint is open.

## 4. Deploy the UI

Push to GitHub, then at share.streamlit.io point a new app at `ui/app.py`. Under
**Advanced settings → Secrets**:

```toml
BACKEND_URL = "https://enterprise-rag-api-xxxxx.run.app"
API_KEY = "the same value you put in Secret Manager"
LOGFIRE_TOKEN = "optional"
```

`ui/app.py` reads `st.secrets` first and the environment second, so the same file
runs locally against `.env` and hosted against secrets with no changes.

---

## Operating notes

**Cold starts.** Scale-to-zero means the first request after idle pays for
container start, the NeMo rails compile, and the FlashRank model download — 30-60
seconds. `--min-instances 1` removes it and costs roughly $8/month. For a demo you
are showing someone live, warm it with a `/health` call a minute beforehand.

**Cost ceiling.** Every component is free-tier at portfolio traffic. The real risk
is not the platform bill but your Groq and Gemini quotas being drained through an
unprotected endpoint, which is the whole reason `API_KEY` exists.

**Rolling back.** `gcloud run services update-traffic enterprise-rag-api --to-revisions=PREVIOUS=100`.

**Logs.** `gcloud run services logs read enterprise-rag-api --region us-central1`.
Every response carries a `request_id`, and it appears on the matching Logfire span
— that is the thread to pull when a user reports a specific bad answer.

---

## Known gaps

Worth stating plainly rather than discovering during an interview:

| Gap | Impact | Fix if it mattered |
|---|---|---|
| Single instance | Throughput ceiling, memory lost on restart | Persistent checkpointer |
| Shared API key | No per-user identity, no revocation | Real auth (OAuth / JWT) |
| No rate limiting | One client can exhaust the quota | `slowapi`, or Cloud Armor |
| Ingestion is manual | New documents need a local CLI run | Cloud Scheduler → Cloud Run job |
| No CI | Tests run only when you remember | GitHub Actions on push |
