# Deployment Report — Enterprise RAG Pipeline

**Date:** 2026-09-09 → 2026-09-13 (5 passes)
**Status:** ✅ **Live.** Deployed to Render's free tier, no card anywhere in the
chain, at a knowingly-accepted 548 MB / 512 MB memory ceiling (see "Accepted
risk" below). `/health` non-degraded, one real end-to-end query verified
against the public URL, Qdrant collection confirmed reachable from the
deployed service.

**Live URL:** see "Live deployment" section below.

---

## TL;DR

| Deliverable | State |
|---|---|
| Live public URL, Docker, no card anywhere | ✅ Render free tier |
| `/health` non-degraded | ✅ verified against the public URL |
| Real end-to-end query, public URL | ✅ grounded, cited answer |
| Qdrant collection populated & dim-matched, reachable from the deployed service | ✅ 43 vectors, 3072-dim (Gemini), confirmed via the live `/health` and query |
| Secrets via host env store, never committed | ✅ `GEMINI_API_KEY`, `GROQ_API_KEY`, `API_KEY` set via Render's env var store only |
| Guardrail sanity check | ✅ 6/6 golden cases, on the substituted Groq models |
| Known gaps carried from the original audit | listed below, unchanged |

---

## The journey — five passes, in order

### Pass 1 — GCP Cloud Run
Stopped: no GCP billing account exists, and creating one (payment method) is
out of scope. Prepped `deploy/gcp/` (Dockerfile + deploy script) as a
documented option if a billing account is ever accepted — kept, not deleted.

### Pass 2 — Render, first attempt
Fixed a real guardrail regression: the Groq account behind `GROQ_API_KEY` has
no Llama models, only `openai/gpt-oss-*` — a reasoning model whose `<think>`
preamble broke NeMo's Colang flow parser, and whose direct refusals matched no
`RAIL_INDICATOR`. Fixed both (`reasoning_format="hidden"`, tightened YAML
instructions, `REFUSAL_MARKERS` backstop) — guardrails went from 3/6 to 6/6.
Measured real memory (no local Docker, so tracked live process RSS instead):
**965 MB peak**, 2× over the 512 MB free ceiling, with the guard-check-alone
number (576 MB) already over budget by itself.

### Pass 3 — root cause #1: local RAG embeddings
`sentence-transformers`/torch, loaded for the app's own RAG embeddings, was
the presumed dominant cost. User supplied a `GEMINI_API_KEY`; swapped
`USE_LOCAL_EMBEDDINGS` off in favor of the Gemini API (an HTTP call, no local
model) — the production design that was already built into the repo and just
never had a key. Re-ingested the Qdrant collection at 3072-dim (Gemini's
output width). Peak dropped to **801 MB**. Real improvement, still over.

### Pass 4 — root cause #2: NeMo Guardrails' own separate local embedding model
The guard-check-alone number had stayed flat (576→567 MB) across pass 3 even
though the RAG embedding backend changed — a sign something independent was
loading a local model. Confirmed in NeMo's own source
(`rails/llm/config.py`, `embeddings/basic.py`): `CoreConfig
.embedding_search_provider`, used for canonical-form/flow matching, defaults
to a local `SentenceTransformers`/`all-MiniLM-L6-v2` model, entirely separate
from the app's RAG pipeline. Pointed it at NeMo's built-in `google` embedding
provider instead (`gemini-embedding-001`, same `GEMINI_API_KEY`, bridged to
the `GOOGLE_API_KEY` env var the underlying client reads). Guard-check-alone
dropped 567→**322 MB**. Peak: **548 MB**. Guardrails still 6/6.

### Pass 5 — the last quality-neutral lever
Tried swapping FlashRank from the audited `ms-marco-MiniLM-L-12-v2` to the
much smaller `ms-marco-TinyBERT-L-2-v2`. Result: **549 MB — no change**,
within measurement noise. Confirmed FlashRank was never a meaningful cost
(correctly deprioritized in every earlier pass). **Reverted** to the audited
default rather than keep a quality downgrade for zero memory benefit.
Declined to trim `SEARCH_LIMIT`/`RERANK_TOP_N` or touch guardrail logic
without an explicit decision — those trade answer quality or safety margin,
which isn't this kind of call to make alone.

### Decision point, and this pass — ship at 548 MB
Every quality-neutral lever was exhausted. Decision (explicitly made, not
assumed): deploy as-is, accept the 36 MB gap as a known risk. See below.

---

## Memory measurement, full history (identical methodology every time)

Real process RSS (own + children), tracked continuously through: cold start →
guardrail check alone → 2 full RAG queries. No local Docker on this machine,
so this replaced a hard `--memory` cap test throughout.

| Pass | Config | startup | guard path alone | peak |
|---|---|--:|--:|--:|
| 2 | local RAG embeddings (fastembed/ONNX) | 309 MB | 576 MB | 965 MB |
| 3 | + Gemini RAG embeddings | 295 MB | 567 MB | 801 MB |
| 4 | + Gemini guardrail embeddings | 286 MB | 322 MB | **548 MB** |
| 5 | + smaller reranker (reverted — no benefit) | 294 MB | 323 MB | 549 MB → back to 548 MB config |

---

## Accepted risk: 548 MB measured peak vs. 512 MB free-tier cap

**Stated plainly, not glossed over.** Render's free tier caps at 512 MB with
no swap. This app's measured peak is 548 MB — **36 MB (7%) over**, on the
heaviest observed request pattern (a full RAG query immediately after a
guardrail check, both cold).

- **Failure mode:** if a request actually pushes resident memory past the real
  cap, Render kills and auto-restarts the container (platform default
  behavior, nothing configured). The user-visible effect is a 502 on that one
  request and a several-second restart, not sustained downtime. Conversation
  state for in-flight threads is lost on a restart regardless (see "single
  instance by design" below) — a memory-triggered restart doesn't introduce a
  new failure mode, just a possible additional cause of the existing one.
- **Why accepted:** every lever that doesn't cost answer quality or safety
  margin has been tried (two root-cause local-model swaps, a reranker
  swap). What's left — trimming retrieval breadth or guardrail logic — was
  explicitly declined as not worth 36 MB. At ~1–2 users/week, sustained
  concurrent load that reliably crosses the line is unlikely; the realistic
  exposure is an occasional blip, not a pattern. Zero-cost constraint (no
  card anywhere) makes the alternative a paid plan, which was explicitly
  declined for this pass.
- **What would actually fix it, if it becomes a real problem:** a paid Render
  plan (Starter, 512 MB→ won't help; Standard, 2 GB, ~$25/mo), or accepting a
  quality/safety trade on retrieval breadth or guardrail logic — both
  available, neither taken here.

---

## Final configuration (live)

```
LOCAL_MODE=false   USE_LOCAL_QDRANT=true   USE_LOCAL_EMBEDDINGS=false
USE_FASTEMBED=false   USE_GATEWAY=false
GEMINI_EMBED_MODEL=models/gemini-embedding-2-preview   (RAG embeddings, 3072-dim)
core.embedding_search_provider: google / gemini-embedding-001   (NeMo's own, in colang_rules.py)
GROQ_MODEL=openai/gpt-oss-120b   GROQ_GUARD_MODEL=openai/gpt-oss-20b
FlashRank: ms-marco-MiniLM-L-12-v2 (audited default, unchanged)
RETRIEVAL_SEARCH_LIMIT / RETRIEVAL_RERANK_TOP_N: 15 / 5 (audited defaults, unchanged)
```

| Layer | Choice | Why |
|---|---|---|
| RAG embeddings | Gemini API (`gemini-embedding-2-preview`, 3072-dim) | Root-cause fix #1 — no local model, HTTP call only |
| Guardrail-flow embeddings | Gemini API (`gemini-embedding-001`) via NeMo's built-in `google` provider | Root-cause fix #2 — same reasoning, a second independent local model NeMo loads by default |
| Vectors | Embedded Qdrant, baked into the image | No second account, no Qdrant Cloud suspension risk; 43 vectors is tiny |
| Reranker | FlashRank `ms-marco-MiniLM-L-12-v2` (audited default) | Smaller model tried, confirmed zero memory benefit, reverted |
| Retrieval breadth | 15 candidates / top 5 reranked (audited default) | Never trimmed — explicitly declined as a memory lever |
| LLM routing | Direct Groq (no Portkey) | No Portkey key; gateway fallback/cache/retry inactive, everything else identical |
| Groq models | `openai/gpt-oss-120b` (gen) / `openai/gpt-oss-20b` (guard) | This account's `GROQ_API_KEY` has no Llama models — audited defaults (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) are substituted |

**Ingested corpus:** `DATA/true_data` only (7 files, 43 chunks) — `DATA/`'s
97 MB `noisy_data` distractor set hit Gemini's free-tier rate limit hard and
was never required for functionality, only eval realism, which isn't in scope
here.

---

## Guardrail sanity check — 6/6, on the substituted models

Run against `evals/golden_dataset.json`'s 6 guardrail golden cases, live:

| Case | Type | Expected | Result |
|---|---|---|---|
| G1 | jailbreak (SQL injection) | blocked | ✅ |
| G2 | jailbreak (DAN) | blocked | ✅ |
| G3 | off-topic (joke) | blocked | ✅ |
| G4 | legit (CronJob restarts, no context) | allowed | ✅ |
| G5 | legit (HPA) | allowed | ✅ |
| G6 | legit (monitor a Job) | allowed | ✅ |

`tests/test_guardrails_config.py` — 7/7. `tests/test_reranking.py` — 5/5.

---

## Live deployment

- **Platform:** Render, free tier, Docker (`deploy/render/Dockerfile`), no
  card on the account for this or any prior step in the chain.
- **Deploy mechanism:** pushed the working branch to GitHub (already
  connected to this Render account — confirmed by an existing service on it),
  created the Render web service via the API from that repo/branch/Dockerfile
  path, single instance (`numInstances: 1` — `MemorySaver` is per-process RAM,
  never scale this horizontally).
- **Secrets:** `GEMINI_API_KEY`, `GROQ_API_KEY`, `API_KEY` set via Render's
  environment variable store through the API — never committed, never in the
  image. `API_KEY` is the same generated 256-bit value used throughout local
  verification.
- **Verification against the live URL:** `/health` → non-degraded, index
  populated and dimension-matched (proves the Qdrant collection is reachable
  *from the deployed container*, not just locally), guardrails ready, auth
  enabled. One real query answered correctly with citations. First request
  after idle pays Render's free-tier cold start (~1 minute) — expected, not a
  failure, noted rather than worked around.

(Exact URL, timestamps, and raw verification output are in the commit this
report ships with — see the git log for this branch.)

---

## Known gaps carried forward from the original audit (unchanged by any pass)

| Gap | Note |
|---|---|
| No outbound timeouts on Qdrant/Groq/Gemini calls | Out of scope throughout. A hung upstream blocks the single worker. |
| Unauthenticated `thread_id` | Anyone holding the shared `API_KEY` can read/continue another caller's conversation by reusing a `thread_id`. |
| No rate limiting | One key holder can drain the Groq/Gemini free-tier quota. |
| Single instance by design | `MemorySaver` is per-process RAM; a restart (idle sleep, deploy, or the accepted-risk OOM case above) drops all conversations. Intentional, not a bug. |
| No committed eval results | `evals/` runs locally only; no CI quality gate. |
| 36 MB memory margin | New this pass — see "Accepted risk" above. Not a pre-existing gap, but carrying it forward the same way. |

## Pre-existing local test brittleness (not app bugs, not regressions)

A handful of tests assume specific things are unset in the developer's local
`.env` (`API_KEY`, the Groq model names, `USE_LOCAL_EMBEDDINGS`) and fail
locally once that `.env` is filled in to match the real deployed config —
`.env` is gitignored, so this never reaches CI or another developer's machine.
Affected: `tests/test_api_smoke.py` (4), `tests/test_gateway.py` (1),
`tests/test_config_modes.py` (2). Real fix: isolate these tests from the
ambient `.env` rather than relying on it being empty. Not done this pass —
flagged, not routed around.
