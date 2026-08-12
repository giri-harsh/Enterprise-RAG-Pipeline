"""
Eval-suite logic that runs without an LLM.

Tool detection and the guardrail confusion matrix decide what your eval numbers
say. A bug here does not crash anything — it just reports a score that is wrong,
which is worse. Both are pure functions, so both are cheap to pin down.
"""

import pytest

from evals.pipeline import detect_tool, _contexts_from_sources
from evals.guardrails_eval import compute_guardrails_metrics, _is_blocked


# ── Tool detection ────────────────────────────────────────────────────────────

def test_detects_guardrail_block():
    assert detect_tool(["Intent: Guardrails Fired", "Retrieval: Skipped"]) == "guardrails"


def test_detects_retrieval():
    assert detect_tool(["Intent: Technical", "Search Term: pod autoscaling"]) == "retrieve_documents"


def test_detects_retrieval_from_the_context_line():
    """The retriever's plan entry now carries counts — detection must still match."""
    assert detect_tool(["Context Retrieved: 5 chunks from 2 source(s)"]) == "retrieve_documents"


def test_detects_conversational():
    assert detect_tool(["Intent: Conversational/Memory", "Retrieval: Skipped"]) == "direct_answer"


def test_guardrails_wins_over_other_signals():
    """A blocked query is blocked, whatever else appears in the trace."""
    steps = ["Intent: Guardrails Fired", "Intent: Technical", "Context Retrieved: 3 chunks"]
    assert detect_tool(steps) == "guardrails"


def test_empty_trace_is_unknown():
    assert detect_tool([]) == "unknown"


# ── Source flattening ─────────────────────────────────────────────────────────

def test_contexts_extracted_from_chunk_dicts():
    sources = [
        {"content": "text one", "source": "a.pdf"},
        {"content": "text two", "source": "b.pdf"},
    ]
    assert _contexts_from_sources(sources) == ["text one", "text two"]


def test_legacy_string_sources_still_load():
    """Eval runs saved before the API returned chunk dicts must remain readable."""
    assert _contexts_from_sources(["plain text"]) == ["plain text"]


# ── Guardrail confusion matrix ────────────────────────────────────────────────

def test_perfect_classifier_scores_one():
    results = [
        {"result": "TP"}, {"result": "TP"},
        {"result": "TN"}, {"result": "TN"},
    ]
    m = compute_guardrails_metrics(results)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["accuracy"] == 1.0


def test_false_negative_lowers_recall_only():
    """A missed jailbreak is a recall failure — precision is untouched."""
    m = compute_guardrails_metrics([{"result": "TP"}, {"result": "FN"}, {"result": "TN"}])
    assert m["recall"] == 0.5
    assert m["precision"] == 1.0


def test_false_positive_lowers_precision_only():
    """Blocking a legitimate question is a precision failure."""
    m = compute_guardrails_metrics([{"result": "TP"}, {"result": "FP"}, {"result": "TN"}])
    assert m["precision"] == 0.5
    assert m["recall"] == 1.0


def test_no_positives_does_not_divide_by_zero():
    m = compute_guardrails_metrics([{"result": "TN"}, {"result": "TN"}])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["accuracy"] == 1.0


def test_empty_results_are_safe():
    m = compute_guardrails_metrics([])
    assert m["total"] == 0
    assert m["accuracy"] == 0.0


# ── Blocked detection ─────────────────────────────────────────────────────────

def test_blocked_flag_is_preferred():
    assert _is_blocked({"blocked": True, "thought_process": []}) is True
    assert _is_blocked({"blocked": False, "thought_process": ["Intent: Guardrails Fired"]}) is False


def test_falls_back_to_trace_when_flag_absent():
    assert _is_blocked({"thought_process": ["Intent: Guardrails Fired"]}) is True
    assert _is_blocked({"thought_process": ["Intent: Technical"]}) is False
