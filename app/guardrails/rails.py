from typing import TYPE_CHECKING, Optional

import logfire

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS

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

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=GUARD_MODEL,
        temperature=0,  # classification, not generation — sampling only adds variance
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

        logfire.info("Guardrails passed.")
        return False, None
