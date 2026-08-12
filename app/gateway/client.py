from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings

# Groq's OpenAI-compatible endpoint, used only when the gateway is bypassed.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# Gateway routing policy.
#
# fallback   Portkey walks `targets` in order and moves to the next one when the
#            current target errors. Groq's free tier rate-limits under load, so
#            the 70B degrading to the 8B is the difference between a slower
#            answer and no answer.
# cache      Identical prompts are served from Portkey without an LLM call.
#            "simple" is an exact-match cache; "semantic" matches near-duplicate
#            prompts but needs a paid plan, and silently downgrades to simple on
#            the free tier — hence "simple" here, stated honestly.
# retry      429 and 503 are transient and worth retrying in place before paying
#            the quality cost of dropping to the smaller model. A 400 or 401 will
#            fail identically every time, so those are not retried.
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503],
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/{settings.GROQ_GUARD_MODEL}"}},
    ],
}


_portkey_client = None
_direct_client = None


def get_portkey_client():
    """
    Client for the responder's generation call.

    Returns the native Portkey client normally, or a plain OpenAI client pointed
    at Groq when USE_GATEWAY is off. Both expose `.chat.completions.create()`, so
    the responder is unchanged either way.

    Direct mode exists for local development and demos. Setting up the gateway
    means creating two virtual keys in the Portkey dashboard, which is a lot of
    ceremony to ask of someone who just wants to run the thing once. What it costs
    is everything the gateway provides — the 70B→8B fallback, the response cache
    and the retry policy are all gateway features, so in direct mode they simply
    do not happen. `/health` reports which mode is live rather than leaving that
    ambiguous.

    Built lazily and cached. Constructing either at module scope made
    `import app.gateway` open an HTTP client as a side effect of the import
    statement, which could raise before any application code ran.
    """
    global _portkey_client, _direct_client

    if not settings.USE_GATEWAY:
        if _direct_client is None:
            from openai import OpenAI

            if not settings.GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is required in direct mode.")
            _direct_client = _DirectGroq(
                OpenAI(api_key=settings.GROQ_API_KEY, base_url=GROQ_BASE_URL),
                settings.GROQ_MODEL,
            )
        return _direct_client

    if _portkey_client is None:
        if not settings.PORTKEY_API_KEY:
            raise RuntimeError(
                "PORTKEY_API_KEY is not set.\n"
                "  Set it, or run locally with LOCAL_MODE=true (calls Groq directly)."
            )
        _portkey_client = Portkey(api_key=settings.PORTKEY_API_KEY, config=GATEWAY_CONFIG)
    return _portkey_client


class _DirectGroq:
    """
    Adapter giving the plain OpenAI client the same call shape as Portkey.

    Portkey takes the model from GATEWAY_CONFIG's targets, so the responder calls
    `create()` without naming one. A direct client needs the model passed
    explicitly — this injects it, so the two are interchangeable at the call site.
    """

    def __init__(self, client, model: str):
        self._client = client
        self._model = model
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        kwargs.setdefault("model", self._model)
        return self._client.chat.completions.create(**kwargs)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    A Portkey-backed ChatOpenAI — drop-in for ChatGroq inside LangChain nodes.

    Why ChatOpenAI rather than ChatGroq, which is what actually serves the tokens:

    Portkey is a proxy exposing an OpenAI-compatible endpoint at
    PORTKEY_GATEWAY_URL. ChatGroq talks to Groq's API directly and offers no way
    to point it at a proxy, so routing through the gateway with it is impossible.
    ChatOpenAI accepts base_url (the gateway) and default_headers (gateway auth
    plus the routing config above). The @slug/model syntax is Portkey's own —
    Groq's client would reject it.

    The models are still Groq's. Portkey sits in the middle, which is what buys
    the fallback, the cache and the per-feature usage attribution.

    `feature` is attached as metadata so the Portkey dashboard can separate
    planner traffic from responder traffic — useful when working out which node
    is burning the token budget.

    When USE_GATEWAY is off, this points ChatOpenAI straight at Groq's own
    OpenAI-compatible endpoint instead. Same class, different base URL — which is
    the whole reason ChatOpenAI was the right choice here in the first place.
    """
    if not settings.USE_GATEWAY:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is required in direct mode.")
        return ChatOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            model=settings.GROQ_MODEL,
            temperature=0,
        )

    if not settings.PORTKEY_API_KEY:
        raise RuntimeError(
            "PORTKEY_API_KEY is not set.\n"
            "  Set it, or run locally with LOCAL_MODE=true (calls Groq directly)."
        )

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
        temperature=0,  # planning is classification — determinism is the point
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=GATEWAY_CONFIG,
            metadata={"feature": feature, "_user": "rag-system"},
        ),
    )


def extract_cache_status(response) -> str:
    """
    Read x-portkey-cache-status off a native Portkey response.

    The header is the only way to know a response came from the gateway cache
    rather than the model. The SDK has moved where it keeps the raw HTTP response
    between versions, so several attribute paths are tried.

    Returns "MISS" when the header is absent. Defaulting the other way would make
    the UI claim cache hits that never happened.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
