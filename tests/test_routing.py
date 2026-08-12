"""
Graph routing.

route_planner decides whether a turn costs a vector search plus a rerank or
nothing at all. It used to route by comparing state["current_query"] against the
literal string "CONVERSATIONAL", which meant control flow depended on the content
of a model-generated string. These tests pin the current behaviour: routing reads
the typed `intent` field and nothing else.
"""

import pytest

from app.agents.graph import route_planner


def test_conversational_intent_skips_retrieval():
    state = {"intent": "conversational", "current_query": "hello there"}
    assert route_planner(state) == "responder"


def test_technical_intent_goes_to_retriever():
    state = {"intent": "technical", "current_query": "how do I autoscale pods"}
    assert route_planner(state) == "retriever"


def test_query_text_cannot_influence_routing():
    """
    The old bug, made explicit. A rewritten search query that happens to contain
    — or exactly equal — the word CONVERSATIONAL must still be retrieved on.
    """
    state = {"intent": "technical", "current_query": "CONVERSATIONAL"}
    assert route_planner(state) == "retriever"

    state = {"intent": "conversational", "current_query": "kubernetes networking"}
    assert route_planner(state) == "responder"


def test_missing_intent_defaults_to_retrieval():
    """Fail toward documentation: an unclassified turn is better answered with
    sources than from the model's own memory."""
    assert route_planner({"current_query": "anything"}) == "retriever"


def test_unexpected_intent_value_defaults_to_retrieval():
    assert route_planner({"intent": "something_new", "current_query": "x"}) == "retriever"


def test_graph_compiles_with_a_checkpointer():
    """The compiled graph must carry a checkpointer, or thread_id memory is a no-op."""
    from app.agents.graph import rag_agent

    assert rag_agent.checkpointer is not None

    nodes = set(rag_agent.get_graph().nodes)
    assert {"planner", "retriever", "responder"} <= nodes
