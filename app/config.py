import os
from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """
    Environment configuration, read once at import.

    Deliberately a plain class rather than pydantic-settings: nothing here needs
    validation or coercion, and required-field enforcement lives where the value
    is actually used, so the app can start with an unset LOGFIRE_TOKEN but fails
    loudly the moment it tries to reach Qdrant without an endpoint.
    """

    # ── Run mode ──────────────────────────────────────────────────────────────
    # LOCAL_MODE swaps the three managed services for local equivalents so the
    # app runs from a single Groq API key:
    #
    #   Qdrant Cloud     → embedded Qdrant, on-disk under QDRANT_LOCAL_PATH
    #   Gemini           → sentence-transformers all-mpnet-base-v2, 768-dim
    #   Portkey gateway  → direct Groq calls
    #
    # This is a development and demo convenience, not the production path. It
    # exercises the same graph, the same guardrails and the same reranker — what
    # changes is where the vectors live and who serves the tokens. Anything that
    # depends on the gateway specifically (fallback, response cache) is inert in
    # this mode, and the UI says so rather than implying otherwise.
    LOCAL_MODE = _flag("LOCAL_MODE")

    # Individual overrides, for mixing modes — cloud Qdrant with local embeddings,
    # say. Each defaults to whatever LOCAL_MODE implies.
    USE_LOCAL_QDRANT = _flag("USE_LOCAL_QDRANT", str(LOCAL_MODE))
    USE_LOCAL_EMBEDDINGS = _flag("USE_LOCAL_EMBEDDINGS", str(LOCAL_MODE))
    USE_GATEWAY = _flag("USE_GATEWAY", str(not LOCAL_MODE))

    # ── Embeddings ────────────────────────────────────────────────────────────
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-2-preview")
    LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "all-mpnet-base-v2")

    # ── Vector DB ─────────────────────────────────────────────────────────────
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_rag")
    # Embedded Qdrant writes here. Same client library and same query API as the
    # cloud service — it is the storage backend that differs, not the code path.
    QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", ".qdrant_local")

    # ── Reasoning engine (Groq) ───────────────────────────────────────────────
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_GUARD_MODEL = os.getenv("GROQ_GUARD_MODEL", "llama-3.1-8b-instant")

    # ── LLM gateway (Portkey) ─────────────────────────────────────────────────
    # The slugs are Portkey virtual keys, created in the Portkey dashboard and
    # each holding its own Groq credential. Two exist so the fallback target uses
    # a different upstream key from the primary — if the fallback shared a key
    # with the primary, a rate limit would take out both and the fallback would
    # be decorative.
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG = os.getenv("PORTKEY_PRIMARY_SLUG", "rag")      # @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = os.getenv("PORTKEY_FALLBACK_SLUG", "brag")  # @brag/llama-3.1-8b-instant

    # ── API surface ───────────────────────────────────────────────────────────
    API_KEY = os.getenv("API_KEY")

    # ── Observability ─────────────────────────────────────────────────────────
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "enterprise-rag")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    @classmethod
    def describe_mode(cls) -> dict:
        """Which backend serves each layer. Surfaced by /health and the UI."""
        return {
            "vectors": "embedded (local disk)" if cls.USE_LOCAL_QDRANT else "Qdrant Cloud",
            "embeddings": (
                f"local · {cls.LOCAL_EMBED_MODEL} · 768-dim"
                if cls.USE_LOCAL_EMBEDDINGS
                else f"Gemini · {cls.GEMINI_EMBED_MODEL} · 3072-dim"
            ),
            "llm": "Portkey gateway → Groq" if cls.USE_GATEWAY else "Groq (direct)",
            "reranker": "FlashRank (local, always)",
            "guardrails": f"NeMo → Groq {cls.GROQ_GUARD_MODEL}",
        }


settings = Settings()

# LangChain reads tracing config from the environment rather than from any object
# we pass it, so these have to be set as env vars. Only enabled when a key exists —
# LangSmith otherwise logs a warning on every single call.
if settings.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGSMITH_TRACING
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
