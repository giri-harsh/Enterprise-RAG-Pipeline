import os
import tempfile
import time
import logfire
from flashrank import Ranker, RerankRequest

# Loaded on first use, not at import. The model is an ONNX file that FlashRank
# downloads and caches on first run; doing that at import time would slow every
# process start (including the ingestion CLI, which never reranks) and would run
# before logfire.configure() has had a chance to set up tracing.
_ranker = None

# Cache directory for the FlashRank ONNX model.
#
# Cloud Run guarantees only /tmp is writable in a read-only container, so the
# original code hardcoded that path. On Windows /tmp doesn't exist, and on any
# platform the path is pre-created by the OS — using tempfile.gettempdir() gives
# the right location on every OS:
#   Linux / macOS / Cloud Run → /tmp/flashrank
#   Windows                   → %TEMP%\flashrank  (e.g. C:\Users\<user>\AppData\Local\Temp\flashrank)
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "flashrank")


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        logfire.info("Loading FlashRank cross-encoder (ms-marco-MiniLM-L-12-v2, ONNX).")
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            _ranker = Ranker(cache_dir=_CACHE_DIR)
        except Exception as exc:
            logfire.warning(f"Could not use {_CACHE_DIR} ({exc}) — falling back to default cache.")
            _ranker = Ranker()
    return _ranker


def rerank_documents(query: str, documents: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-score retrieved chunks against the query with a cross-encoder.

    Why a second ranking stage exists at all:

    Qdrant ranks by cosine similarity between the query embedding and document
    embeddings. Those document embeddings were computed at ingestion time, with no
    knowledge of any query — that is precisely what makes them indexable, and also
    what makes them approximate. Two passages about "scaling pods" and "scaling
    nodes" land close together in that space whether or not either answers the
    question actually asked.

    A cross-encoder takes the query and one passage as a single concatenated
    input, so its attention runs across both at once and it scores genuine
    relevance rather than embedding proximity. The cost is that nothing can be
    precomputed — it is one model forward pass per candidate, which is why it is
    used on 15 candidates and never on the whole collection.

    FlashRank makes that affordable by running a small quantised MiniLM as ONNX on
    CPU: a few milliseconds per passage, no GPU, no network call, no per-token
    billing. The alternative — Cohere Rerank — is more accurate but adds an API
    dependency, latency, and cost to every single query.

    Takes and returns the full chunk dicts (content, source, score) so source
    attribution survives the ranking step.
    """
    if not documents:
        return []

    start = time.monotonic()

    try:
        ranker = _get_ranker()

        # FlashRank matches on 'id' and 'text'. Index into the original list so
        # each result can be mapped back to its full record, metadata intact.
        passages = [{"id": i, "text": doc["content"]} for i, doc in enumerate(documents)]

        results = ranker.rerank(RerankRequest(query=query, passages=passages))

        reranked = []
        for res in results[:top_n]:
            original = documents[res["id"]]
            reranked.append({**original, "rerank_score": round(float(res["score"]), 4)})

        logfire.info(
            "Reranking complete",
            duration_s=round(time.monotonic() - start, 3),
            candidates=len(documents),
            kept=len(reranked),
            top_score=reranked[0]["rerank_score"] if reranked else None,
        )
        return reranked

    except Exception as exc:
        # Degrade, do not fail. A dead reranker costs answer quality; raising here
        # would cost the user their answer entirely. Qdrant's cosine ordering is a
        # perfectly serviceable second-best.
        logfire.error(f"Reranking failed, falling back to vector order: {exc}")
        return documents[:top_n]
