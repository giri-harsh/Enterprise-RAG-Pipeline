import time
import logfire

from app.config import settings

BATCH_SIZE = 50
GEMINI_DIM = 3072
LOCAL_DIM = 768  # all-mpnet-base-v2

_active_model = None
_model_type: str | None = None  # "gemini" or "local"


# ── Model initialisation ───────────────────────────────────────────────────────

def _load_gemini():
    """
    Build the Gemini embedder and verify it with one real call.

    The probe matters. Constructing GoogleGenerativeAIEmbeddings succeeds with an
    invalid key — the failure only surfaces on first use, which would otherwise be
    halfway through indexing a corpus. One embed_query call up front turns that
    into an immediate, clear failure.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBED_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
    )
    model.embed_query("probe")
    logfire.info(f"Gemini embeddings ready ({settings.GEMINI_EMBED_MODEL}, {GEMINI_DIM}-dim).")
    return model


def _load_local():
    """
    Load the local sentence-transformers model.

    Not in requirements-prod.txt — it pulls PyTorch (~2 GB) and times out the
    container build. Production is Gemini-only, so reaching this in a deployed
    container is a configuration error rather than a recoverable fallback.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Local embeddings requested but sentence-transformers is not installed.\n"
            "  Local demo:  pip install sentence-transformers\n"
            "  Deployed:    this means GEMINI_API_KEY is missing or invalid — "
            "check the secret binding."
        ) from exc

    logfire.info(f"Loading local embeddings ({settings.LOCAL_EMBED_MODEL}, {LOCAL_DIM}-dim).")
    return SentenceTransformer(settings.LOCAL_EMBED_MODEL)


def _init():
    """
    Pick and load the embedding model once per process, on first use.

    Explicit configuration decides this, not a silent fallback. An earlier version
    tried Gemini and quietly dropped to the local model when the probe failed —
    which produced 768-dim vectors while the collection, the README and the docs
    all said 3072. Every subsequent search then failed on dimension mismatch, with
    nothing in the logs pointing at the cause.

    Now USE_LOCAL_EMBEDDINGS decides, and a Gemini failure raises.
    """
    global _active_model, _model_type
    if _active_model is not None:
        return

    if settings.USE_LOCAL_EMBEDDINGS:
        _active_model = _load_local()
        _model_type = "local"
        return

    if not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "  Set it, or run in local mode with LOCAL_MODE=true "
            "(uses sentence-transformers instead)."
        )

    try:
        _active_model = _load_gemini()
        _model_type = "gemini"
    except Exception as exc:
        raise RuntimeError(
            f"Gemini embeddings are unavailable: {exc}\n"
            "  Check GEMINI_API_KEY and that GEMINI_EMBED_MODEL names a real model.\n"
            "  To run locally without Gemini, set LOCAL_MODE=true and re-ingest "
            "with --wipe (the vector dimension changes from 3072 to 768)."
        ) from exc


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """
    Vector width for the active model.

    Fixed at collection creation and not changeable afterwards — Qdrant rejects a
    query whose dimensions do not match the collection. Switching embedding models
    therefore requires re-ingesting with --wipe.
    """
    _init()
    return GEMINI_DIM if _model_type == "gemini" else LOCAL_DIM


def get_model_type() -> str:
    _init()
    return _model_type


# ── Batch embedding with retry ─────────────────────────────────────────────────

def _embed_batch(batch: list[str]) -> list[list[float]]:
    if _model_type != "gemini":
        return _active_model.encode(batch, show_progress_bar=False).tolist()

    # Exponential backoff on rate limits: 1s → 2s → 4s → 8s. Gemini's free tier
    # limits requests per minute, and a corpus of any size will hit it — retrying
    # is the difference between a slow ingest and a failed one.
    for attempt in range(4):
        try:
            return _active_model.embed_documents(batch)
        except Exception as exc:
            message = str(exc).lower()
            transient = any(x in message for x in ("429", "rate", "quota", "resource_exhausted"))
            if transient and attempt < 3:
                wait = 2**attempt
                logfire.warning(f"Gemini rate limit — retrying in {wait}s (attempt {attempt + 1}/4).")
                time.sleep(wait)
            else:
                logfire.error(f"Gemini embedding failed: {exc}")
                raise

    raise RuntimeError("Gemini rate limit persisted after 4 attempts.")


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    _init()
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        with logfire.span("Embed batch", model=_model_type, start=i, size=len(batch)):
            out.extend(_embed_batch(batch))
    return out
