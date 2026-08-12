# ==============================================================================
# logfire.configure() must run before any app module is imported.
#
# Instrumentation patches libraries at import time. Any module that loads before
# configure() holds references to the unpatched versions, and its spans never
# reach the collector — which shows up as a trace with mysterious gaps rather
# than as an error. Hence the unusual import order below.
# ==============================================================================
import os
import time
import uuid

import logfire
from dotenv import load_dotenv

load_dotenv()

from app.observability import configure_logfire

# Tracing is optional. configure_logfire falls back to local-only mode when
# LOGFIRE_TOKEN is blank rather than raising, which is what actually makes the
# token optional — passing token=None here used to crash the API at startup.
TRACING_ENABLED = configure_logfire("enterprise-rag-api")

# Safe to import application modules now.
from contextlib import asynccontextmanager
from typing import Optional, Literal

from fastapi import FastAPI, Response, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.graph import rag_agent
from app.guardrails import initialize_rails, guard


API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Build the NeMo rails once at startup.

    LLMRails compiles the Colang flows and embeds every example utterance for
    canonical-form matching. That takes seconds — acceptable once per process,
    unacceptable per request. Done here rather than in the deprecated
    @app.on_event("startup"), which FastAPI has scheduled for removal.
    """
    logfire.info("Starting up — initialising guardrails.")
    initialize_rails()
    yield

    # Close the vector client explicitly. Embedded Qdrant holds an exclusive lock
    # on its data directory and releases it in __del__ — which runs during
    # interpreter shutdown, when it may no longer be able to import what it needs.
    # That leaves the lock behind and makes the next run fail with "already
    # accessed by another instance". Closing here avoids that entirely.
    try:
        from app.services.retrieval import qdrant_service

        if qdrant_service._client is not None:
            qdrant_service._client.close()
            qdrant_service._client = None
    except Exception as exc:
        logfire.warning(f"Vector client did not close cleanly: {exc}")

    logfire.info("Shutting down.")


app = FastAPI(
    title="Enterprise Agentic RAG API",
    version="1.0.0",
    description="Guardrailed, agentic RAG over enterprise documentation.",
    lifespan=lifespan,
)

# CORS is opt-in via ALLOWED_ORIGINS. The Streamlit UI calls this server-side and
# needs none, but a browser front-end would be blocked without it. No wildcard
# default — that would make a deployed instance callable from any page on the web.
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Shared-secret check on the query endpoint.

    Every /query call spends tokens against your Groq, Gemini and Qdrant quotas.
    A public unauthenticated URL is therefore a standing invitation to drain them,
    and the bill is not theoretical.

    If API_KEY is unset the check is skipped, so local development needs no setup.
    Startup logs a warning when that happens — see the /health payload.
    """
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


class QueryRequest(BaseModel):
    q: str = Field(min_length=1, max_length=2000, description="The user's question.")
    thread_id: str = Field(
        default="default_user",
        max_length=128,
        description="Conversation key. Same value = same memory.",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    thought_process: list[str]
    status: str
    sources: list[dict]
    blocked: bool
    latency_ms: int
    request_id: str


@app.get("/", tags=["meta"])
def home():
    return {"service": "Enterprise Agentic RAG API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health():
    """
    Readiness probe.

    Reports whether the rails actually compiled, rather than just whether the
    process is alive — a container that answers on the port but has no guardrail
    is worse than one that is plainly down, because it fails open.
    """
    from app.guardrails import rails as rails_module
    from app.config import settings
    from app.services.retrieval.qdrant_service import collection_stats

    rails_ready = rails_module._rails is not None
    index = collection_stats()

    # An empty collection is the single most common reason a local demo returns
    # nothing useful, and it produces no error anywhere — retrieval just comes
    # back empty. Surfacing it here turns a confusing session into an obvious one.
    indexed = bool(index.get("exists") and index.get("vectors"))

    if rails_ready and indexed:
        state = "ok"
    elif rails_ready:
        state = "degraded"
    else:
        state = "unhealthy"

    return {
        "status": state,
        "guardrails": "ready" if rails_ready else "not initialised",
        "index": index,
        "mode": settings.describe_mode(),
        "auth": "enabled" if API_KEY else "disabled",
        "cors_origins": ALLOWED_ORIGINS or "none",
    }


@app.get("/graph", tags=["meta"])
def get_graph_image():
    """Render the compiled LangGraph workflow as a PNG."""
    try:
        return Response(
            content=rag_agent.get_graph().draw_mermaid_png(),
            media_type="image/png",
        )
    except Exception as exc:
        logfire.warning(f"Graph render failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph rendering unavailable (requires network access to the Mermaid renderer).",
        )


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["rag"],
    dependencies=[Depends(require_api_key)],
)
def query(request: QueryRequest):
    """
    Run one turn through the guardrail gate and the agent graph.

    Guardrails sit in front of the graph rather than inside it as a node. A
    blocked query then costs one small-model classification and nothing else — no
    embedding, no vector search, no rerank, no 70B call. The tradeoff is that the
    gate cannot see conversation state, so it judges each message alone.
    """
    request_id = str(uuid.uuid4())[:8]
    started = time.monotonic()

    with logfire.span(
        "POST /query",
        request_id=request_id,
        thread_id=request.thread_id,
        question=request.q[:120],
    ):
        # Gate 1 — guardrails.
        try:
            rail_fired, rail_response = guard(request.q)
        except Exception as exc:
            # Fail closed. A broken guardrail must not become an open pipe to the
            # LLM; refusing the request is the safe direction to fail in.
            logfire.error(f"Guardrail evaluation failed: {exc}", request_id=request_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Safety checks are unavailable. Request refused.",
            )

        if rail_fired:
            logfire.info("Blocked by guardrails.", request_id=request_id)
            return QueryResponse(
                question=request.q,
                answer=rail_response,
                thought_process=["Intent: Guardrails Fired", "Retrieval: Skipped"],
                status="Blocked by guardrails.",
                sources=[],
                blocked=True,
                latency_ms=int((time.monotonic() - started) * 1000),
                request_id=request_id,
            )

        # Gate 2 — agent graph. Invoked synchronously so Logfire's context
        # variables propagate; the async path loses span nesting.
        try:
            result = rag_agent.invoke(
                {
                    "messages": [{"role": "user", "content": request.q}],
                    "intent": "technical",
                    "current_query": request.q,
                    "documents": [],
                    "plan": [],
                    "status": "Starting.",
                },
                config={"configurable": {"thread_id": request.thread_id}},
            )
        except Exception as exc:
            # 500, not a 200 with an apology in the body. Returning success on
            # failure hides outages from every client, dashboard and alert that
            # reads the status code — which is all of them.
            logfire.error(f"Agent execution failed: {exc}", request_id=request_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query processing failed. Request ID: {request_id}",
            )

        return QueryResponse(
            question=request.q,
            answer=result.get("final_answer") or "",
            thought_process=result.get("plan") or [],
            status=result.get("status") or "",
            sources=result.get("documents") or [],
            blocked=False,
            latency_ms=int((time.monotonic() - started) * 1000),
            request_id=request_id,
        )
