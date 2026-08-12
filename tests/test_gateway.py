"""
Portkey gateway configuration and response parsing.

The gateway config is a plain dict that is never type-checked and never fails
loudly — a typo in the strategy mode or a broken target slug produces a working
client that quietly has no fallback. These assertions make the intended shape
explicit.
"""

import pytest

from app.gateway.client import GATEWAY_CONFIG, extract_cache_status


def test_fallback_strategy_is_configured():
    assert GATEWAY_CONFIG["strategy"]["mode"] == "fallback"


def test_two_targets_exist_in_priority_order():
    """
    Fallback needs somewhere to fall back to. Order matters: Portkey tries targets
    top-down, so the larger model must be first.
    """
    targets = GATEWAY_CONFIG["targets"]
    assert len(targets) == 2

    primary = targets[0]["override_params"]["model"]
    secondary = targets[1]["override_params"]["model"]
    assert "70b" in primary.lower()
    assert "8b" in secondary.lower()


def test_targets_use_portkey_slug_syntax():
    """@slug/model is Portkey-specific routing — a bare model name silently 404s."""
    for target in GATEWAY_CONFIG["targets"]:
        assert target["override_params"]["model"].startswith("@")


def test_retries_only_on_transient_status_codes():
    """
    429 and 503 are worth retrying. A 400 or 401 will fail identically every time,
    so retrying them just multiplies latency.
    """
    retry = GATEWAY_CONFIG["retry"]
    assert retry["attempts"] >= 1
    assert set(retry["on_status_codes"]) == {429, 503}


def test_caching_is_enabled():
    assert GATEWAY_CONFIG["cache"]["mode"] in {"simple", "semantic"}


# ── Cache status parsing ──────────────────────────────────────────────────────

class _Response:
    def __init__(self, attr_name=None, headers=None):
        if attr_name:
            setattr(self, attr_name, type("Raw", (), {"headers": headers or {}})())


def test_reads_cache_hit_header():
    resp = _Response("_raw_response", {"x-portkey-cache-status": "HIT"})
    assert extract_cache_status(resp) == "HIT"


def test_header_value_is_normalised_to_uppercase():
    resp = _Response("_raw_response", {"x-portkey-cache-status": "hit"})
    assert extract_cache_status(resp) == "HIT"


def test_alternate_attribute_paths_are_tried():
    """The SDK has moved this attribute between versions, hence the defensive loop."""
    assert extract_cache_status(_Response("_response", {"x-portkey-cache-status": "HIT"})) == "HIT"
    assert extract_cache_status(_Response("_http_response", {"x-portkey-cache-status": "HIT"})) == "HIT"


def test_missing_header_reports_miss():
    """Unknown must read as MISS, never as HIT — the UI would claim a false cache hit."""
    assert extract_cache_status(_Response("_raw_response", {})) == "MISS"
    assert extract_cache_status(_Response()) == "MISS"
