# Deployment Report — Enterprise RAG Pipeline

**Date:** 2026-09-09
**Goal:** local-only → publicly reachable, low-traffic hosted deployment (Cloud Run).
**Status:** ⚠️ **Blocked on one credential.** Everything that does not require a
cloud account is done, verified, and committed. The deploy itself is one command
(`deploy/deploy.sh`) once GCP access exists.

---

## TL;DR

| Deliverable | State |
|---|---|
| Live public URL (Cloud Run) | ❌ **blocked** — no GCP credentials available on this machine; cannot create an account / attach billing (out of scope per stop rule) |
| `/health` non-degraded + real query | ✅ verified **locally** (see below); will re-verify against the public URL post-deploy |
| Qdrant collection populated & dim-matched | ✅ `enterprise_rag`, 22 vectors, 768-dim, matches `all-mpnet-base-v2` |
| Secrets via GCP Secret Manager | ✅ scripted in `deploy/deploy.sh` (`--set-secrets`), nothing hardcoded/committed |
| This report | ✅ |

---

## The blocker (why the deploy did not complete)

Cloud Run requires the Google Cloud SDK, an authenticated identity, and a
billing-enabled project. On this machine:

- `gcloud` **is not installed** (nor `docker`).
- **No GCP credential of any kind** is present — no `gcloud` auth state, no
  Application Default Credentials, no service-account JSON, nothing in `.env`.
- Creating a GCP account and attaching a billing method is an account-creation +
  payment-method + spend action, and `gcloud auth login` is an interactive
  browser OAuth flow (typically 2FA). All of these are explicit stop conditions.

Installing `gcloud` alone does not help — it still needs an identity and a
billing account behind it. I did not find a free FaaS alternative that avoids the
account-creation wall either (Render/Fly/HF Spaces all need a new signup, Fly
needs a card up front, and Render's free tier is 512 MB — below the 2 GiB the
FlashRank ONNX model needs).

### What unblocks it

Provide **either**:

1. **Preferred —** a GCP project id with billing enabled, plus `gcloud auth login`
   completed in a shell on this machine (or a service-account key JSON with
   `roles/run.admin`, `roles/cloudbuild.builds.editor`, `roles/secretmanager.admin`,
   `roles/artifactregistry.admin`). Then: `GCP_PROJECT=<id> bash deploy/deploy.sh`.
2. An alternative host you already have an account on (Render paid ≥1 GB, Fly.io,
   Hugging Face Spaces Docker). The image in `deploy/Dockerfile.cloudrun` is
   host-portable (`$PORT` aware); only the secret-wiring in `deploy/deploy.sh` is
   Cloud-Run-specific.

The deploy target stays exactly as scoped: `--max-instances 1`,
`--min-instances 0`, 2 GiB / 2 vCPU, `--timeout 300`, `--allow-unauthenticated`,
single worker. No load balancer, no multi-instance (conversation state is
per-process `MemorySaver` RAM by design).

---

## Chosen configuration and why

**Mode: fully local backends in the container** — embedded Qdrant + local
`sentence-transformers` embeddings + direct Groq. Flags baked into
`deploy/Dockerfile.cloudrun`:

```
LOCAL_MODE=true  USE_LOCAL_QDRANT=true  USE_LOCAL_EMBEDDINGS=true  USE_GATEWAY=false
```

Rationale (per-layer, cheapest/simplest that needs no new account):

| Layer | Decision | Why not the cloud option |
|---|---|---|
| Vectors | **Embedded Qdrant**, `.qdrant_local` baked into image (298 KB, 22 vectors) | Qdrant Cloud key *is* present and the cluster is alive, **but its collection list is empty** — it would need re-ingestion anyway, and the free cluster carries a suspension-from-inactivity risk that then needs a keepalive workflow. Embedded removes all of that. Index is immutable and tiny; re-ingest + redeploy to update it. |
| Embeddings | **local `all-mpnet-base-v2`** (768-dim) | `GEMINI_API_KEY` is absent. Google AI Studio keys are free/no-card, but adding local embeddings to the image avoids the extra credential entirely. Cost: `torch` CPU wheel + ~420 MB model weights pre-baked at build → larger image, slightly longer cold start. Acceptable at ~1–2 users/week with `--timeout 300`. |
| LLM routing | **direct Groq** | `PORTKEY_API_KEY` is absent. Gateway only adds fallback/cache/retry; the graph, guardrails and reranker are identical without it. |
| Groq models | `GROQ_MODEL=openai/gpt-oss-120b`, `GROQ_GUARD_MODEL=openai/gpt-oss-20b` | **The Groq account behind `GROQ_API_KEY` does not have `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`** (the repo defaults). Its 14 available models are gpt-oss / qwen3 / compound family. Picked the 120B for generation and 20B for the guardrail classifier. Set as `--set-env-vars`, and added to `.env` locally. |

`requirements-prod.txt` still targets the Gemini path (unchanged except the two
pins below); the container Dockerfile layers `sentence-transformers`/`torch` on
top. If you would rather deploy the Gemini path, supply `GEMINI_API_KEY`, build
with the repo-root `Dockerfile`, and re-ingest against Qdrant Cloud with `--wipe`
(vector width becomes 3072).

---

## Work completed and verified

### 1. Dependency pins (audit risk #2)
`requirements-prod.txt`, the two deps flagged `# Pin before deploying`, pinned to
the versions resolved in the working venv (Python 3.12.8):

- `openai==2.54.0`
- `langchain-google-genai==4.4.0`

### 2. Qdrant collection (deliverable #3)
`scripts/preflight.py` (run with `PYTHONUTF8=1` — it crashes on cp1252 consoles,
noted as a minor gap): **Ready**, all layers green after the model overrides.
Collection `enterprise_rag` — **22 vectors, 768-dim, Cosine**, dimension matches
the active embedding model. Corpus is `DATA/true_data` (7 source docs); the 97 MB
`DATA/noisy_data` distractor set is intentionally not indexed.

### 3. `/health` verification (deliverable #2, local)
Clean run against `app.main:app`:

```json
{"status":"ok","guardrails":"ready",
 "index":{"exists":true,"collection":"enterprise_rag","vectors":22,"dimension":768,"distance":"Cosine"},
 "mode":{"vectors":"embedded (local disk)","embeddings":"local · all-mpnet-base-v2 · 768-dim",
         "llm":"Groq (direct)","reranker":"FlashRank (local, always)",
         "guardrails":"NeMo → Groq openai/gpt-oss-20b"},
 "auth":"enabled","cors_origins":"none"}
```

NeMo rails compiled, index present and populated, **not degraded**.

### 4. End-to-end query (deliverable #2, local)
`POST /query {"q":"How do I autoscale pods in Kubernetes?","thread_id":"smoke-1"}`
→ 200, grounded multi-step answer with inline `[1]` citations from the indexed
Kubernetes docs. Guardrails pass, retrieval + FlashRank rerank + generation all
execute.

### 5. Auth (deliverable #4)
Generated a 256-bit `API_KEY` (`python -c 'secrets.token_hex(32)'`), written to
`.env` only (confirmed `git check-ignore` positive — never committed).
`deploy/deploy.sh` pushes it and `GROQ_API_KEY` to Secret Manager via
`gcloud secrets` and wires them with `--set-secrets` — **no plain env-var secrets**.
Verified locally: `POST /query` with no key → **401**, with the key → **200**,
`/health` reports `"auth":"enabled"`.

### 6. Deploy artifacts (ready to run)
- `deploy/Dockerfile.cloudrun` — the embedded-path image.
- `deploy/deploy.sh` — enable APIs → create/rotate secrets → Cloud Build →
  `gcloud run deploy` with the exact scoped flags → curl `/health` + smoke query.

### 7. Qdrant keepalive workflow (audit risk #1)
**Not added — deliberately.** The keepalive is only needed "if using Qdrant
Cloud". The chosen config uses embedded Qdrant, so there is no cluster to keep
warm. `.github/workflows/` currently has only `ci.yml`. If you switch to the
Qdrant Cloud path, add a daily-cron workflow that GETs
`${QDRANT_CLUSTER_ENDPOINT}/collections` with the `api-key` header — a stub is in
the "If you switch to Qdrant Cloud" note above.

---

## Known gaps carried over from the audit (unchanged by this pass)

| Gap | Status |
|---|---|
| **No outbound timeouts** on Qdrant/Gemini/Groq/Portkey calls | Not fixed — explicitly out of scope for this pass. A slow/hung upstream can block the single worker. Fix: add `timeout=` to every httpx/requests client in `app/services/` and `app/gateway/`. |
| **Unauthenticated `thread_id`** | Any caller with the shared `API_KEY` can read/continue another caller's conversation by guessing/reusing a `thread_id`. Acceptable at portfolio traffic; real fix is per-user identity (OAuth/JWT). |
| **No rate limiting** | One client with the key can drain the Groq quota. Fix: `slowapi` or Cloud Armor. |
| **Single instance by design** | Intentional — `MemorySaver` is per-process RAM. A restart drops all conversations. Documented, not a bug. Fix if it ever matters: `PostgresSaver`/`RedisSaver` (same interface). |
| **No committed eval results** | `evals/` runs locally only; no CI gate on answer quality. |
| **`preflight.py` crashes on cp1252 consoles** | Minor. Run with `PYTHONUTF8=1` (or `set PYTHONUTF8=1`). Fix: `sys.stdout.reconfigure(encoding="utf-8")` at the top of the script. |
| **`/query` answer text shows mojibake for em-dashes** (`â€"`) | Cosmetic encoding issue in generation output handling; pre-existing, not deployment-related. |

---

## Files changed this pass

- `requirements-prod.txt` — pinned `openai`, `langchain-google-genai`.
- `deploy/Dockerfile.cloudrun` — new.
- `deploy/deploy.sh` — new.
- `DEPLOYMENT_REPORT.md` — this file.
- `.env` (gitignored, not committed) — added `GROQ_MODEL`, `GROQ_GUARD_MODEL`
  overrides and the generated `API_KEY`.

## Next action for you

1. Get me GCP access (see "What unblocks it") — **or** tell me which host you have
   an account on.
2. Then: `GCP_PROJECT=<your-project> bash deploy/deploy.sh` — it deploys,
   wires secrets, and prints the live URL + `/health` + a smoke query.
3. I'll re-verify `/health` (`"auth":"enabled"`, non-degraded) against the public
   URL and run one real query to close deliverables #1 and #2.
