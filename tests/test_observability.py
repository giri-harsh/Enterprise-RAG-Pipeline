"""
Logfire configuration.

Tracing is documented as optional — LOGFIRE_TOKEN can stay blank. That was only
true by accident: three of the four entry points called logfire.configure() in a
way that raised without a token, taking down the API at startup and the ingestion
CLI before it indexed anything. The token was nominally optional and actually
required, and nothing caught it because nothing tested the no-token path.
"""

import os

import pytest

from app.observability import configure_logfire


def test_blank_token_does_not_raise(monkeypatch):
    """The regression. A missing token must degrade, not crash."""
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    assert configure_logfire("test-service", console=False) is False


def test_empty_string_token_is_treated_as_absent(monkeypatch):
    """`LOGFIRE_TOKEN=` in a .env file yields "", not None."""
    monkeypatch.setenv("LOGFIRE_TOKEN", "")
    assert configure_logfire("test-service", console=False) is False


def test_whitespace_token_is_treated_as_absent(monkeypatch):
    """A stray space after `=` in .env should not be read as a credential."""
    monkeypatch.setenv("LOGFIRE_TOKEN", "   ")
    assert configure_logfire("test-service", console=False) is False


def test_spans_still_work_without_a_token(monkeypatch):
    """
    Local-only mode has to leave the API intact. Every module in this codebase
    calls logfire.span() and logfire.info() unconditionally — if those become
    invalid when tracing is off, disabling tracing breaks the application.
    """
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    configure_logfire("test-service", console=False)

    import logfire

    with logfire.span("probe", attribute="value"):
        logfire.info("message", key="value")
        logfire.warning("warning")
        logfire.error("error")


def test_a_broken_token_does_not_take_the_app_down(monkeypatch):
    """Observability failing is not a reason for the application to fail."""
    monkeypatch.setenv("LOGFIRE_TOKEN", "obviously-not-a-real-token")
    result = configure_logfire("test-service", console=False)
    assert isinstance(result, bool)


def test_console_default_does_not_hit_the_fallback(monkeypatch, capsys):
    """
    console=True must not be forwarded to logfire.configure().

    Its `console` parameter takes ConsoleOptions or False; a bare True raises
    `'bool' object has no attribute 'span_style'`. The helper's own except branch
    caught that and carried on, so the only visible symptom was a stray
    "[observability] Logfire setup failed" line — easy to miss, and it meant the
    default path was silently running through error handling.
    """
    monkeypatch.delenv("LOGFIRE_TOKEN", raising=False)
    capsys.readouterr()

    configure_logfire("test-service")  # console defaults to True

    assert "Logfire setup failed" not in capsys.readouterr().out


def test_every_entry_point_uses_the_helper():
    """
    Guard against the pattern coming back. Each entry point previously rolled its
    own logfire.configure(), and they drifted — one crashed on a blank token,
    another survived only because it happened to sit inside a try/except.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    entry_points = [
        root / "app" / "main.py",
        root / "app" / "ingestion" / "processor.py",
        root / "evals" / "app.py",
        root / "ui" / "app.py",
    ]

    for path in entry_points:
        source = path.read_text(encoding="utf-8")
        assert "configure_logfire" in source, (
            f"{path.name} does not use configure_logfire(). Calling "
            "logfire.configure() directly reintroduces the blank-token crash."
        )
        assert "logfire.configure(token=" not in source, (
            f"{path.name} calls logfire.configure(token=...) directly. "
            "token=None raises when LOGFIRE_TOKEN is unset."
        )
