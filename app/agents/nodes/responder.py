import logfire

from app.agents.state import AgentState
from app.gateway import get_portkey_client, extract_cache_status

# Groq's on-demand tier is billed and rate-limited on tokens per minute, so the
# prompt has a hard ceiling. Roughly 4 chars per token puts 25k chars near 6k
# tokens, which leaves room for the model's own output inside the window.
MAX_CONTEXT_CHARS = 25_000


def _format_history(messages: list[dict]) -> str:
    lines = []
    for msg in messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_context(documents: list[dict]) -> tuple[str, list[str]]:
    """
    Concatenate chunks up to the character budget, labelling each with its source.

    Labelling matters: without a filename attached to each block, the model has no
    way to attribute a claim, and any citation it produces is invented. Numbering
    the blocks gives it something concrete to point at.
    """
    parts, used_sources, total = [], [], 0

    for i, doc in enumerate(documents, start=1):
        block = f"[{i}] Source: {doc['source']}\n{doc['content']}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            logfire.warning(
                "Context truncated at the token budget",
                included=len(parts),
                dropped=len(documents) - len(parts),
            )
            break
        parts.append(block)
        used_sources.append(doc["source"])
        total += len(block)

    return "\n\n".join(parts), used_sources


def generate_node(state: AgentState) -> dict:
    """
    Produce the final answer.

    Uses the native Portkey client rather than the LangChain wrapper for one
    specific reason: LangChain normalises the provider response into its own
    message object and discards the raw HTTP headers along the way. The gateway
    reports cache hits in `x-portkey-cache-status`, so reading that header — and
    surfacing "Cache: Hit" in the UI — requires the unwrapped response.
    """
    history_str = _format_history(state["messages"])
    user_msg = state["messages"][-1]["content"] if state["messages"] else ""
    documents = state.get("documents") or []

    if state.get("intent") == "conversational":
        logfire.info("Answering from conversation history.")
        cited_sources = []
        prompt = f"""You are an enterprise IT assistant. Answer the user's latest
message using the conversation history below. Be brief and direct.

CONVERSATION HISTORY:
{history_str}

LATEST MESSAGE:
"{user_msg}"
"""
    elif not documents:
        # Retrieval found nothing. Say so rather than letting the model fill the
        # silence — an unsupported answer is worse than an admitted gap, and this
        # is the failure mode faithfulness scoring is meant to catch.
        logfire.warning("No context retrieved — answering with an explicit gap.")
        cited_sources = []
        prompt = f"""You are an enterprise IT assistant. A documentation search for
the question below returned no relevant material.

Tell the user plainly that the documentation does not cover this, and suggest how
they might rephrase. Do not answer from general knowledge and do not speculate.

USER QUESTION:
"{user_msg}"
"""
    else:
        logfire.info("Answering from retrieved documentation.")
        context, cited_sources = _build_context(documents)
        prompt = f"""You are a senior technical architect answering from internal
documentation.

Rules:
- Use only the TECHNICAL CONTEXT below. Do not add outside knowledge.
- Cite the block number inline, like [2], for every factual claim.
- If the context does not cover part of the question, say which part is missing.

TECHNICAL CONTEXT:
{context}

CONVERSATION HISTORY:
{history_str}

USER QUESTION:
"{user_msg}"
"""

    with logfire.span("LLM synthesis"):
        try:
            response = get_portkey_client().chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = response.choices[0].message.content

            cache_hit = extract_cache_status(response) == "HIT"
            if cache_hit:
                logfire.info("Gateway cache hit — served without an LLM call.")
                plan = state["plan"] + ["Cache: Hit"]
                status = "Cache hit — instant response."
            else:
                plan = state["plan"]
                status = "Response generated."

            unique_sources = sorted(set(cited_sources))
            if unique_sources:
                plan = plan + [f"Sources: {', '.join(unique_sources)}"]

            return {
                "final_answer": content,
                "status": status,
                "plan": plan,
                "messages": [{"role": "assistant", "content": content}],
            }

        except Exception as exc:
            # Re-raise. main.py owns the HTTP error contract; swallowing the
            # exception here would hide a gateway outage behind a 200 OK.
            logfire.error(f"LLM generation failed: {exc}")
            raise
