from typing import TypedDict, List, Annotated, Literal
import operator


# The planner classifies every incoming turn into exactly one of these.
# Kept as a Literal rather than a bare str so a typo in a node is caught by the
# type checker instead of silently routing to the wrong branch at runtime.
Intent = Literal["conversational", "technical"]


class AgentState(TypedDict):
    """
    Shared state passed between graph nodes.

    LangGraph merges each node's returned dict into this state. By default a
    returned key *replaces* the existing value. `messages` is the exception:
    Annotated[..., operator.add] registers operator.add as its reducer, so
    returning a one-element list appends rather than overwrites. That is what
    makes conversation history accumulate across turns instead of being reset by
    whichever node ran last.
    """

    # Reduced with operator.add — nodes return only their new messages.
    messages: Annotated[List[dict], operator.add]

    # The planner's routing decision. Held in its own field rather than encoded
    # into current_query, so that control flow never depends on the *content* of
    # a user-derived string.
    intent: Intent

    # For a technical turn this is the planner's rewritten search query, which is
    # what actually gets embedded. For a conversational turn it is the user's
    # message unchanged. Purely data — never used for routing.
    current_query: str

    # Reranked chunks with their source filenames attached, ready for citation.
    documents: List[dict]

    # Human-readable trace of the decisions taken, surfaced in the UI and parsed
    # by the eval suite's tool detector.
    plan: List[str]

    status: str
    final_answer: str
