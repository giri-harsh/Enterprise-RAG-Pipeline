import logfire
from qdrant_client import QdrantClient

from app.config import settings
from app.services.retrieval.embedding import embed_query


_client = None


def get_client() -> QdrantClient:
    """
    Qdrant client, built on first use and reused afterwards.

    Two backends, one interface. `path=` runs Qdrant embedded against a local
    directory — same client library, same query API, same collection semantics,
    just no network and no account. That is what makes the local demo possible
    without weakening what is being demonstrated: the retrieval code below does
    not know or care which one it is talking to.

    The embedded backend takes an exclusive lock on its directory, so exactly one
    process can hold it. The API server owns it; the ingestion CLI must not be
    running at the same time.

    Constructed lazily rather than at module scope so that importing this module
    does not open a connection as a side effect.
    """
    global _client
    if _client is None:
        if settings.USE_LOCAL_QDRANT:
            logfire.info(f"Using embedded Qdrant at {settings.QDRANT_LOCAL_PATH}")
            _client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
        else:
            if not settings.QDRANT_URL:
                raise RuntimeError(
                    "QDRANT_CLUSTER_ENDPOINT is not set.\n"
                    "  Set it, or run locally with LOCAL_MODE=true "
                    "(embedded Qdrant, no account needed)."
                )
            _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return _client


def collection_stats() -> dict:
    """Collection state for the preflight check and /health. Never raises."""
    try:
        client = get_client()
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            return {"exists": False, "collection": settings.QDRANT_COLLECTION}

        info = client.get_collection(settings.QDRANT_COLLECTION)
        return {
            "exists": True,
            "collection": settings.QDRANT_COLLECTION,
            "vectors": info.points_count,
            "dimension": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
        }
    except Exception as exc:
        return {"exists": False, "error": str(exc)}


def search_enterprise_knowledge(query: str, limit: int = 8) -> list[dict]:
    """
    Embed the query and pull the nearest chunks by cosine similarity.

    Returns chunk dicts carrying content, source filename, source_type and the
    similarity score. The source fields matter: they are what lets the responder
    cite a real filename instead of inventing one.

    Returns an empty list on failure rather than raising. The responder handles an
    empty context explicitly — it tells the user the documentation does not cover
    the question — which is a better outcome than a 500 for what is often a
    transient network blip.
    """
    try:
        query_vector = embed_query(query)

        response = get_client().query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "content": point.payload.get("text", ""),
                "source": point.payload.get("source", "unknown"),
                "source_type": point.payload.get("source_type", "unknown"),
                "score": round(float(point.score), 4),
            }
            for point in response.points
        ]

    except Exception as exc:
        # A dimension mismatch lands here too, and it is the most likely cause in
        # practice: it means the collection was indexed with one embedding model
        # and is being queried with another. embedding.py logs that case directly.
        logfire.error(f"Qdrant search failed: {exc}", query=query[:80])
        return []
