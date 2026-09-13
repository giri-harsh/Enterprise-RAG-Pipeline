import os
from typing import TYPE_CHECKING, Optional

import logfire

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    YAML_CONTENT,
    RAIL_INDICATORS,
    REFUSAL_MARKERS,
)

if TYPE_CHECKING:  # pragma: no cover
    from nemoguardrails import LLMRails


# Guardrail intent classification runs on the small model, not the 70B.
#
# The gate answers one narrow question — is this message on-topic and non-adversarial
# — and it runs on every single request before anything else happens. Spending the
# large model on it would put its latency and its token cost on the critical path of
# every query, including the ones that get blocked and never reach retrieval.
GUARD_MODEL = settings.GROQ_GUARD_MODEL

_rails: Optional["LLMRails"] = None


def initialize_rails() -> None:
    """
    Build the LLMRails singleton. Called once from the FastAPI lifespan handler.

    nemoguardrails and langchain_groq are imported here rather than at module
    scope. Both are heavy — NeMo alone pulls a large transitive tree — and
    importing them at load time meant that anything touching this package, the
    test suite included, paid for the whole stack just to read a config constant.
    Building the rails is a startup-time operation, so the import belongs at
    startup time too.
    """
    global _rails

    from langchain_groq import ChatGroq
    from nemoguardrails import RailsConfig, LLMRails

    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set — the guardrail LLM cannot be built.")

    # The YAML config below points NeMo's own canonical-form/flow search at its
    # built-in Google embedding provider (see colang_rules.py) instead of the
    # local torch default. That provider's underlying google-genai client reads
    # GOOGLE_API_KEY from the environment — this app names the same credential
    # GEMINI_API_KEY, so bridge it here rather than duplicate the secret or
    # rename the app's own setting. setdefault: never overwrite an operator's
    # own GOOGLE_API_KEY if one is already set.
    if settings.GEMINI_API_KEY:
        os.environ.setdefault("GOOGLE_API_KEY", settings.GEMINI_API_KEY)
    else:
        raise RuntimeError(
            "GEMINI_API_KEY is not set — required for NeMo's guardrail-flow "
            "embedding search (see colang_rules.py's core.embedding_search_provider)."
        )

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=GUARD_MODEL,
        temperature=0,  # classification, not generation — sampling only adds variance
        # gpt-oss is a reasoning model: without this it prefixes every completion
        # with a <think>...</think> block. NeMo's Colang 1.0 flow engine parses the
        # LLM output line-by-line to resolve canonical forms, and the reasoning
        # preamble derails that parsing — no flow matches, the rail never fires,
        # and the raw completion (think tags and all) is returned. "hidden" makes
        # Groq strip the reasoning server-side so NeMo sees a clean completion.
        # The Llama models the rails were originally tuned against had no such
        # preamble. See DEPLOYMENT_REPORT.md.
        reasoning_format="hidden",
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    # Passing llm= overrides any provider declared in the YAML. This is the single
    # source of truth for which model guards the app.
    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"NeMo Guardrails initialised on {GUARD_MODEL}.")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a message through the rails.

    Returns:
        (True, response)  a rail fired — return this text and skip the pipeline
        (False, None)     clean — continue to the graph

    How firing is detected: NeMo's generate() returns only the final assistant
    message, with no field indicating which flow matched or whether one matched at
    all. A rail's canned reply and a genuine model answer arrive through the same
    channel. So the response is substring-matched against RAIL_INDICATORS, which
    holds a distinctive fragment of every `define bot` message. The coupling that
    creates is documented in colang_rules.py and enforced by
    tests/test_guardrails_config.py.
    """
    if _rails is None:
        # Fail closed. An uninitialised gate previously waved everything through,
        # which turns a startup failure into a silently unguarded endpoint.
        raise RuntimeError("Guardrails are not initialised — refusing to process the request.")

    with logfire.span("Guardrails check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        if any(indicator in content for indicator in RAIL_INDICATORS):
            logfire.info("Rail fired.", query=message[:80])
            return True, content

        # gpt-oss path: the guard model refused in its own words instead of
        # routing through a Colang flow. Match its refusal at the start of the
        # completion, apostrophe- and case-normalised.
        head = content.strip().lower().replace("’", "'").replace("`", "'")[:60]
        if any(head.startswith(marker) for marker in REFUSAL_MARKERS):
            logfire.info("Rail fired (direct refusal).", query=message[:80])
            return True, content

        logfire.info("Guardrails passed.")
        return False, None
