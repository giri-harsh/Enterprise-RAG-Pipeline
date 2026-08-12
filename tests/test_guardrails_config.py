"""
Guardrail configuration integrity.

rails.py decides whether a rail fired by substring-matching the response against
RAIL_INDICATORS. That coupling is invisible to the type checker: reword a bot
message in COLANG_CONTENT without updating its indicator, and detection silently
stops working — blocked queries sail through to the RAG pipeline while the
guardrails eval keeps reporting healthy numbers.

These tests exist to make that drift loud.
"""

import pytest

from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


def test_every_indicator_appears_in_a_bot_definition():
    """The core invariant. If this fails, rail detection is already broken."""
    orphans = [i for i in RAIL_INDICATORS if i not in COLANG_CONTENT]
    assert not orphans, (
        f"These RAIL_INDICATORS no longer match any text in COLANG_CONTENT: {orphans}. "
        "A bot message was reworded without updating its indicator."
    )


def test_every_bot_definition_has_an_indicator():
    """The reverse direction: a new rail whose firing nothing can detect."""
    bot_blocks = [
        line.strip()
        for line in COLANG_CONTENT.splitlines()
        if line.strip().startswith("define bot")
    ]
    assert len(bot_blocks) == len(RAIL_INDICATORS), (
        f"{len(bot_blocks)} bot definitions but {len(RAIL_INDICATORS)} indicators. "
        "Every 'define bot' needs a matching entry in RAIL_INDICATORS."
    )


def test_indicators_are_specific_enough():
    """
    A short indicator risks matching a genuine answer and blocking it. These
    phrases are the whole safety mechanism, so they must be distinctive.
    """
    for indicator in RAIL_INDICATORS:
        assert len(indicator) >= 25, f"Indicator too short to be safe: {indicator!r}"


def test_indicators_are_unique():
    assert len(RAIL_INDICATORS) == len(set(RAIL_INDICATORS))


def test_colang_flows_reference_defined_intents():
    """Every 'user X' inside a flow must have a matching 'define user X'."""
    defined = {
        line.strip()[len("define user ") :]
        for line in COLANG_CONTENT.splitlines()
        if line.strip().startswith("define user ")
    }
    assert defined, "No user intents defined at all."

    lines = COLANG_CONTENT.splitlines()
    in_flow = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("define flow"):
            in_flow = True
            continue
        if not stripped:
            in_flow = False
            continue
        if in_flow and stripped.startswith("user "):
            intent = stripped[len("user ") :]
            assert intent in defined, f"Flow references undefined user intent: {intent!r}"


def test_yaml_declares_no_model():
    """
    rails.py injects ChatGroq directly, which overrides any `models:` block. A
    declaration here would be dead config that misstates which model guards the
    app — it previously claimed gpt-3.5-turbo, which was never true.
    """
    assert "models:" not in YAML_CONTENT
    assert "gpt-3.5-turbo" not in YAML_CONTENT


def test_yaml_still_carries_general_instructions():
    assert "instructions:" in YAML_CONTENT
    assert "Kubernetes" in YAML_CONTENT
