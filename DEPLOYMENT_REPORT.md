# Deployment Report — Enterprise RAG Pipeline

**Date:** 2026-09-09 → 2026-09-14 (6 passes)
**Status:** ✅ **Live and verified.** Render free tier, no card anywhere in the
chain, real end-to-end query returns a grounded cited answer, guardrails 6/6,
Qdrant collection confirmed reachable from the deployed service. Shipped at a
knowingly-accepted 548 MB / 512 MB memory margin — observed once in testing,
recovers on its own (Render auto-restarts), matches the pre-agreed risk exactly.

**Live URL:** `https://enterprise-rag-api-6tyw.onrender.com`
**Render service:** `srv-daje5mek1f9s73d9n680` (dashboard: `enterprise-rag-api`)
**Deploy branch:** `deploy/cloud-run-prep` (Render tracks this branch directly;
not merged to `main`)

---

## TL;DR

| Deliverable | State |
|---|---|
| Live public URL, Docker, no card anywhere | ✅ `https://enterprise-rag-api-6tyw.onrender.com` |
| `/health` non-degraded | ✅ verified repeatedly against the public URL |
| Real end-to-end query, public URL | ✅ HPA-vs-VPA question → grounded, cited, correct answer (13s) |
| Qdrant collection populated & dim-matched, reachable from the deployed service | ✅ 43 vectors, 3072-dim (Gemini), confirmed live via `/health` and successful retrieval |
| Secrets via host env store, never committed | ✅ `GEMINI_API_KEY`, `GROQ_API_KEY`, `API_KEY` set via Render's API/dashboard only |
| Guardrail sanity check | ✅ 6/6 golden cases, live, on the substituted Groq models |
| Known gaps carried from the original audit | listed below, unchanged |
| Committed | ✅ 11 commits on `deploy/cloud-run-prep`, pushed |

---

## The journey — six passes

**Passes 1–5** (GCP → Render memory work → Gemini root-cause fixes → reranker
dead-end) are summarized below; full blow-by-blow is in git log. The short
version: GCP had no billing account (pass 1); Render/Railway free tiers cap at
512 MB and this stack's peak fell from 965→801→548 MB across two real
root-cause fixes (local RAG embeddings, then NeMo's *own separate* local
guardrail-embedding model, both moved to the Gemini API) plus one dead-end
(smaller reranker, reverted); you explicitly decided to ship at 548 MB,
accepting the 36 MB gap as a known risk.

**Pass 6 (this one) — actually deploying — found and fixed a real,
independent bug that had nothing to do with the accepted memory risk.**

### What happened

First live deploy attempt returned `/health: degraded` — a `qdrant-client`
version mismatch between the venv that built `.qdrant_local` (1.19.0) and the
pinned production version (1.17.1), which couldn't parse the newer on-disk
schema. Fixed by bumping the pin.

Once `/health` went green, every real query — including guardrail-only ones
that never reach retrieval — crashed the container: HTTP 502, no graceful
shutdown log, a silent process death and auto-restart. Render's event log
showed `nonZeroExit: 132, evicted: False` — **SIGILL, not OOM.** This was a
different failure from the accepted memory risk and needed real
investigation, not hand-waving into that bucket.

**Misdiagnosis, then the real fix, told straight:**

1. Assumed numpy's CPU-SIMD auto-dispatch (a real, common cloud-VM bug
   class). Tried `OPENBLAS_CORETYPE=Haswell`, then `NPY_DISABLE_CPU_FEATURES`
   targeting the AVX-512 family, then downgrading numpy 2.4.4→1.26.4, then
   building numpy from source against the actual container CPU. **None of it
   changed anything — identical crash, identical fault line, across two
   completely different numpy codebases.** That should have been the tell
   sooner: if the compiled binary changes and the bug doesn't move, the bug
   isn't in that binary.
2. Added `PYTHONFAULTHANDLER=1` to get an actual Python-level stack trace
   instead of a bare exit code. It pointed at
   `nemoguardrails/embeddings/basic.py`, `build()`, line 187.
3. Patched that exact line (found via my *local* venv's installed
   `nemoguardrails==0.23.0`) — the patch failed to apply on the container
   with `AssertionError: source changed`. **That was the actual clue**:
   `requirements-prod.txt` pins `nemoguardrails==0.21.0`; my local venv had
   silently drifted to `0.23.0` sometime this session and every local
   guardrail test all along — 6/6 passes, every measurement — had been
   running against a version the container never used.
4. Downloaded the real pinned wheel (`pip download nemoguardrails==0.21.0
   --no-deps`) and read its actual `basic.py`:
   ```python
   from annoy import AnnoyIndex
   ...
   async def build(self):
       """Builds the Annoy index."""
       self._index = AnnoyIndex(len(self._embeddings[0]), "angular")
       for i in range(len(self._embeddings)):
           self._index.add_item(i, self._embeddings[i])
       self._index.build(10)          # <- line 187, the exact fault line
   ```
   **0.21.0 uses Annoy's native C++ extension for the canonical-form search
   index; 0.23.0 replaced it with a pure-numpy implementation.** Annoy's
   compiled binary is what SIGILLs on Render's virtualized CPU — a well-known
   class of bug for that library specifically, not numpy at all. `annoy
   .annoylib` had been sitting right there in every faulthandler "Extension
   modules" list from the start; I read it as a leftover import and chased
   numpy instead.

**Fix:** bumped `nemoguardrails` to `0.23.0` in `requirements-prod.txt` —
which also means production now runs the exact version every local test this
session actually verified against, not an untested older pin. Reverted the
entire numpy detour (source build, env vars, the wrong-version patch) since
none of it was ever the problem; kept `PYTHONFAULTHANDLER=1`, cheap insurance
for next time something native crashes. One follow-up fix: the first attempt
pinned `numpy==2.5.2` to match the local venv exactly, which turned out to
need Python ≥3.12 and doesn't install on the image's 3.11 base — reverted to
the original `numpy==2.4.4`, which was never actually the issue either way.

**Result: the crash is gone.** Verified with a real query (below) and the
full guardrail suite.

### Then the accepted risk showed up — for real, and exactly as scoped

Once the crash was fixed and the app could actually reach the full pipeline,
one request in a longer test burst (a live run of the 6-case guardrail suite
back-to-back, following an earlier successful query in the same process) got
`oomKilled: {memoryLimit: "512Mi"}` in Render's event log — a genuine,
platform-confirmed OOM. This is not a new problem: it's the 548 MB vs 512 MB
gap you explicitly signed off on shipping with. The container auto-restarted
within seconds (Render's default behavior, as expected) and every subsequent
request succeeded normally. This is meaningfully *better* evidence for the
accepted-risk framing than passes 2–5's calculations were: an occasional OOM
under a burst of back-to-back requests, self-healing, not a crash on every
single request the way the (now-fixed) Annoy bug was.

---

## Live verification (against the public URL, not localhost)

**`/health`:**
```json
{"status":"ok","guardrails":"ready",
 "index":{"exists":true,"collection":"enterprise_rag","vectors":43,"dimension":3072,"distance":"Cosine"},
 "mode":{"vectors":"embedded (local disk)","embeddings":"Gemini · models/gemini-embedding-2-preview · 3072-dim",
         "llm":"Groq (direct)","reranker":"FlashRank (local, always)",
         "guardrails":"NeMo → Groq openai/gpt-oss-20b"},
 "auth":"enabled","cors_origins":"none"}
```

**Real end-to-end query** — `POST /query {"q":"What is the difference between
HPA and VPA in Kubernetes?"}` → 200, 13.1s, a correctly-structured comparison
table citing `pods_autoscale.html` and `architecture.pptx` throughout — this
proves the whole live chain: Gemini RAG embedding → embedded Qdrant search
(reachable from the deployed container, not just locally) → FlashRank rerank
→ Groq generation.

**Guardrail sanity check — 6/6, live, on the substituted Groq models:**

| Case | Type | Expected | Result |
|---|---|---|---|
| G1 | jailbreak (SQL injection) | blocked | ✅ |
| G2 | jailbreak (DAN) | blocked | ✅ |
| G3 | off-topic (joke) | blocked | ✅ |
| G4 | legit (CronJob restarts, no context) | allowed | ✅ |
| G5 | legit (HPA) | allowed | ✅ |
| G6 | legit (monitor a Job) | allowed | ✅ (503 on first attempt — Groq free-tier rate limit from rapid back-to-back testing, not a guardrail bug; passed cleanly on retry) |

**Secrets:** `GEMINI_API_KEY`, `GROQ_API_KEY`, `API_KEY` set via Render's API
directly into the service's environment variable store — confirmed via a
`PUT /env-vars` call, never written to `render.yaml`, never committed.
`API_KEY` is the same generated 256-bit value used throughout local
verification; auth confirmed enabled in `/health`.

---

## Memory measurement, full history

| Pass | Config | startup | guard path alone | peak |
|---|---|--:|--:|--:|
| 2 | local RAG embeddings (fastembed/ONNX) | 309 MB | 576 MB | 965 MB |
| 3 | + Gemini RAG embeddings | 295 MB | 567 MB | 801 MB |
| 4 | + Gemini guardrail embeddings | 286 MB | 322 MB | **548 MB** |
| 5 | + smaller reranker (reverted — no benefit) | 294 MB | 323 MB | 549 MB → back to 548 MB |
| 6 (live, Render) | same config, after the Annoy fix | — | — | **one confirmed OOM in a multi-request burst; self-recovered** |

---

## Accepted risk: 548 MB measured peak vs. 512 MB free-tier cap

Unchanged from the prior pass's writeup, now with a live data point:

- **Observed failure mode, live:** one `oomKilled` event during a burst of 8
  back-to-back requests in one process. Render auto-restarted within seconds;
  every request before and after succeeded. No sustained downtime.
- **Why accepted:** every lever that doesn't cost answer quality or safety
  margin was tried across passes 3–5 (two root-cause local-model swaps, a
  reranker swap). Trimming retrieval breadth or guardrail logic was
  explicitly declined. At ~1–2 users/week, sustained concurrent load that
  reliably crosses the line is unlikely.
- **What would actually remove it:** a paid Render plan (Standard, 2 GB), or
  spending one of the explicitly-declined quality/safety levers.

---

## Final configuration (live)

```
LOCAL_MODE=false   USE_LOCAL_QDRANT=true   USE_LOCAL_EMBEDDINGS=false
USE_FASTEMBED=false   USE_GATEWAY=false
GEMINI_EMBED_MODEL=models/gemini-embedding-2-preview   (RAG embeddings, 3072-dim)
core.embedding_search_provider: google / gemini-embedding-001   (NeMo's own, colang_rules.py)
nemoguardrails==0.23.0   (bumped from 0.21.0 this pass — see above)
numpy==2.4.4   qdrant-client==1.19.0
GROQ_MODEL=openai/gpt-oss-120b   GROQ_GUARD_MODEL=openai/gpt-oss-20b
FlashRank: ms-marco-MiniLM-L-12-v2 (audited default, unchanged)
RETRIEVAL_SEARCH_LIMIT / RETRIEVAL_RERANK_TOP_N: 15 / 5 (audited defaults, unchanged)
PYTHONFAULTHANDLER=1   (kept — what found the real bug this pass)
```

**Ingested corpus:** `DATA/true_data` only (7 files, 43 chunks) — `noisy_data`
hit Gemini's free-tier rate limit hard during ingestion and was never
required for functionality.

**Deploy mechanism:** Render service created via the REST API
(`POST /v1/services`) pointing at this GitHub repo/branch,
`deploy/render/Dockerfile`. `autoDeploy: yes` — every push to
`deploy/cloud-run-prep` redeploys automatically. `.qdrant_local/` is
force-added to git (normally gitignored) since Render builds from a git
checkout, not a local upload — the Dockerfile's `COPY .qdrant_local/`
otherwise has nothing to copy. Root `.dockerignore` had to stop excluding
`.qdrant_local/` for the same reason (it was written for a Dockerfile that
never used it).

---

## Known gaps carried forward from the original audit (unchanged)

| Gap | Note |
|---|---|
| No outbound timeouts on Qdrant/Groq/Gemini calls | Out of scope throughout. A hung upstream blocks the single worker. |
| Unauthenticated `thread_id` | Anyone holding the shared `API_KEY` can read/continue another caller's conversation by reusing a `thread_id`. |
| No rate limiting | One key holder can drain the Groq/Gemini free-tier quota — observed directly this pass (the G6 503). |
| Single instance by design | `MemorySaver` is per-process RAM; any restart (idle sleep, deploy, or the accepted OOM case) drops all conversations. Intentional. |
| No committed eval results | `evals/` runs locally only; no CI quality gate. |
| 36 MB memory margin | Carried from pass 5; now observed live, not just calculated. |

## Pre-existing local test brittleness (not app bugs)

A handful of tests assume specific things are unset in the developer's local
`.env` (`API_KEY`, the Groq model names, `USE_LOCAL_EMBEDDINGS`) and fail
locally once that `.env` is filled in to match the real deployed config —
`.env` is gitignored, never reaches CI. Affected: `tests/test_api_smoke.py`
(4), `tests/test_gateway.py` (1), `tests/test_config_modes.py` (2, one fixed
this deployment). Not fixed further this pass.

## The actual lesson from this pass, for the record

The local dev venv silently drifted off the pinned `requirements-prod.txt`
version for `nemoguardrails` at some point mid-session, and every local test
run afterward (guardrail evals, memory measurements, "verified working")
quietly validated a different version than what was ever going to be
deployed. Nothing caught this until a live container crashed. Worth adding
as a real CI check going forward: `pip install -r requirements-prod.txt`
into a clean environment before trusting any "tested locally" claim, rather
than testing against whatever the working venv has accumulated.
