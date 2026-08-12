import logfire

from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

# Cast wide on the vector search, then let the cross-encoder narrow it.
#
# The two stages optimise different things. Qdrant compares a query embedding to
# document embeddings that were computed without ever seeing the query — cheap,
# indexable, and approximate. The cross-encoder reads query and passage together
# in one forward pass, so it can judge actual relevance, but it costs a model
# inference per candidate and cannot be precomputed.
#
# Retrieving 15 and keeping 5 is the usual shape of that tradeoff: enough
# candidates that a relevant chunk ranked 11th by cosine still has a chance,
# few enough that reranking stays in the tens of milliseconds.
SEARCH_LIMIT = 15
RERANK_TOP_N = 5


def retrieve_node(state: AgentState) -> dict:
    """Vector search, then cross-encoder rerank, preserving source attribution."""
    query = state["current_query"]

    with logfire.span("Knowledge retrieval", query=query[:120]):
        candidates = search_enterprise_knowledge(query, limit=SEARCH_LIMIT)
        logfire.info(f"Retrieved {len(candidates)} candidates from Qdrant.")

        if not candidates:
            logfire.warning("Vector search returned nothing — responder will have no context.")
            return {
                "documents": [],
                "status": "No relevant documentation found.",
                "plan": state["plan"] + ["Context: none found"],
            }

        with logfire.span("Semantic reranking"):
            # rerank_documents takes and returns the full candidate dicts, so
            # `source` and the original vector score survive into the response.
            # They previously did not: the node kept only `content`, which meant
            # the API could return text but never say where it came from.
            ranked = rerank_documents(query, candidates, top_n=RERANK_TOP_N)
            logfire.info(f"Reranked to top {len(ranked)} chunks.")

        sources = sorted({d["source"] for d in ranked})
        logfire.info("Context assembled", sources=sources)

    return {
        "documents": ranked,
        "status": f"Found context in {len(sources)} document(s).",
        "plan": state["plan"] + [f"Context Retrieved: {len(ranked)} chunks from {len(sources)} source(s)"],
    }
