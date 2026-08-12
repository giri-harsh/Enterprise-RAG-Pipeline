"""
Run-mode configuration.

LOCAL_MODE swaps three managed services for local equivalents. Getting the flag
logic wrong is quiet and expensive: the app would start, then reach for a service
that is not configured, and fail on the first user query rather than at startup.
"""

import importlib
import os

import pytest


def _settings_with(**env):
    """Reload app.config under a specific environment and return Settings."""
    original = {k: os.environ.get(k) for k in env}
    try:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        import app.config

        importlib.reload(app.config)
        return app.config.Settings
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        import app.config

        importlib.reload(app.config)


_MODE_KEYS = dict(USE_LOCAL_QDRANT=None, USE_LOCAL_EMBEDDINGS=None, USE_GATEWAY=None)


def test_local_mode_flips_all_three_layers():
    s = _settings_with(LOCAL_MODE="true", **_MODE_KEYS)
    assert s.USE_LOCAL_QDRANT is True
    assert s.USE_LOCAL_EMBEDDINGS is True
    assert s.USE_GATEWAY is False  # gateway OFF is the point of local mode


def test_cloud_mode_is_the_default():
    s = _settings_with(LOCAL_MODE="false", **_MODE_KEYS)
    assert s.USE_LOCAL_QDRANT is False
    assert s.USE_LOCAL_EMBEDDINGS is False
    assert s.USE_GATEWAY is True


def test_layers_can_be_overridden_individually():
    """Cloud Qdrant with local embeddings — a real combination while migrating."""
    s = _settings_with(
        LOCAL_MODE="false",
        USE_LOCAL_EMBEDDINGS="true",
        USE_LOCAL_QDRANT=None,
        USE_GATEWAY=None,
    )
    assert s.USE_LOCAL_EMBEDDINGS is True
    assert s.USE_LOCAL_QDRANT is False
    assert s.USE_GATEWAY is True


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "on"])
def test_truthy_spellings_all_work(value):
    assert _settings_with(LOCAL_MODE=value, **_MODE_KEYS).LOCAL_MODE is True


@pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
def test_falsy_spellings_all_work(value):
    assert _settings_with(LOCAL_MODE=value, **_MODE_KEYS).LOCAL_MODE is False


def test_describe_mode_covers_every_layer():
    """/health and the UI both render this — a missing key would show as blank."""
    described = _settings_with(LOCAL_MODE="true", **_MODE_KEYS).describe_mode()
    assert set(described) == {"vectors", "embeddings", "llm", "reranker", "guardrails"}
    assert all(v for v in described.values())


def test_describe_mode_reports_the_active_backends():
    local = _settings_with(LOCAL_MODE="true", **_MODE_KEYS).describe_mode()
    assert "embedded" in local["vectors"]
    assert "768" in local["embeddings"]
    assert "direct" in local["llm"].lower()

    cloud = _settings_with(LOCAL_MODE="false", **_MODE_KEYS).describe_mode()
    assert "Qdrant Cloud" in cloud["vectors"]
    assert "3072" in cloud["embeddings"]
    assert "Portkey" in cloud["llm"]


def test_reranker_and_guardrails_are_local_in_both_modes():
    """Neither depends on the managed services — the demo is not a lesser demo."""
    for mode in ("true", "false"):
        described = _settings_with(LOCAL_MODE=mode, **_MODE_KEYS).describe_mode()
        assert "FlashRank" in described["reranker"]
        assert "NeMo" in described["guardrails"]
