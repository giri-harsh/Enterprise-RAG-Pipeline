from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)


def route_planner(state: AgentState) -> str:
    """
    Send the turn down one of two branches based on the planner's intent.

    Routes on state["intent"], never on state["current_query"]. Those two used to
    be the same field, with the planner writing the sentinel "CONVERSATIONAL" into
    the query slot — which meant a rewritten search query that happened to equal
    that string would have been misrouted, and it left the query field carrying
    two unrelated meanings. Intent is now its own typed field.

    Unknown values fall through to retrieval: answering from documentation is the
    safer default if classification ever returns something unexpected.
    """
    return "responder" if state.get("intent") == "conversational" else "retriever"


workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
    },
)

workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)


# Conversation memory.
#
# MemorySaver keeps checkpoints in the process's own memory, keyed by the
# thread_id passed in config. That is the right choice for a single-instance
# deployment and for local development, and it is why the app needs no database.
#
# The tradeoff is real and worth stating plainly: memory does not survive a
# restart, and it is not shared between instances. Horizontal scaling therefore
# requires either sticky sessions or swapping this for a persistent checkpointer
# (langgraph.checkpoint.postgres.PostgresSaver, same interface). The deployment
# is pinned to a single instance for exactly this reason — see DEPLOYMENT.md.
checkpointer = MemorySaver()

rag_agent = workflow.compile(checkpointer=checkpointer)
