import logfire

from app.agents.state import AgentState
from app.gateway import get_langchain_llm

# Portkey-backed LLM: fallback + cache + retry, same .invoke() interface as ChatGroq.
#
# Built on first call rather than at import. Constructing it opens an HTTP client,
# and doing that at module scope meant importing this node — or anything that
# transitively imports it, including app.agents.graph — performed network setup as
# a side effect of the import statement.
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_langchain_llm(feature="planner")
    return _llm

# What the model is asked to emit when no retrieval is needed. It is a wire
# format between the prompt and this function only — it never reaches the graph
# state or the router, which work off the typed `intent` field instead.
_CONVERSATIONAL_TOKEN = "CONVERSATIONAL"


def _format_history(messages: list[dict]) -> str:
    """Render all turns except the newest as a plain transcript."""
    lines = []
    for msg in messages[:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def planner_node(state: AgentState) -> dict:
    """
    Classify the turn and, when retrieval is needed, rewrite it into a search query.

    Two jobs in one LLM call:

    1. Routing. Greetings, thanks, and follow-ups answerable from the transcript
       alone ("what did I just ask?") skip retrieval entirely. That saves an
       embedding call, a vector search and a rerank on turns where documentation
       cannot help.

    2. Query rewriting. The raw user message is often a poor search key, because
       follow-ups depend on earlier context — "and how do I scale it?" embeds to
       almost nothing useful. The planner resolves those references against the
       history and emits a standalone query. This is what actually gets embedded.
    """
    history = _format_history(state["messages"])
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""You are the planning step of an enterprise documentation assistant.

CONVERSATION HISTORY:
{history}

LATEST MESSAGE:
"{user_message}"

Decide one of two things.

1. If the latest message is a greeting, an acknowledgement, or a question that can
   be answered using ONLY the conversation history above (for example "what did I
   just ask?"), output exactly: {_CONVERSATIONAL_TOKEN}

2. Otherwise it needs documentation. Rewrite it as a standalone search query,
   resolving any pronouns or references against the history so the query makes
   sense on its own. Output only the query text.

Output {_CONVERSATIONAL_TOKEN} or the search query. Nothing else."""

    with logfire.span("Planner decision"):
        raw = _get_llm().invoke(prompt).content.strip()

        # Small models sometimes wrap the sentinel in punctuation or quotes rather
        # than emitting it bare, so compare on a normalised form. Anything that
        # isn't the sentinel is treated as a search query.
        is_conversational = raw.strip("\"'`.! ").upper() == _CONVERSATIONAL_TOKEN

        logfire.info(
            "Intent classified",
            intent="conversational" if is_conversational else "technical",
            raw_output=raw[:120],
        )

    if is_conversational:
        return {
            "intent": "conversational",
            "current_query": user_message,
            "status": "Answering from conversation history.",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"],
        }

    return {
        "intent": "technical",
        "current_query": raw,
        "status": f"Searching documentation for: {raw}",
        "plan": ["Intent: Technical", f"Search Term: {raw}"],
    }
