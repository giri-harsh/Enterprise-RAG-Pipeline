"""
Does the API start at all?

No test imported app.main until now, which is how a startup crash shipped
unnoticed: logfire.configure(token=None) raised whenever LOGFIRE_TOKEN was blank,
so the API died before serving a request. Every unit test passed the whole time,
because none of them touched the module that broke.

These tests import app.main with no credentials set — the state a fresh clone is
in — and check the endpoints that do not need a live backend.
"""

import pytest

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient",
    reason="fastapi not installed; see requirements-test.txt",
)
TestClient = fastapi_testclient.TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    A client with no real credentials and an isolated on-disk vector store.

    TestClient is not used as a context manager on purpose: entering it runs the
    lifespan handler, which builds the NeMo rails and needs a working Groq key.
    These tests are about the module importing and the app object being wired
    correctly, not about the guardrail actually functioning.
    """
    import os

    os.environ["LOCAL_MODE"] = "true"
    os.environ["QDRANT_LOCAL_PATH"] = str(tmp_path_factory.mktemp("qdrant"))
    os.environ.pop("LOGFIRE_TOKEN", None)
    os.environ.pop("API_KEY", None)

    from app.main import app

    return TestClient(app)


def test_importing_app_main_does_not_raise():
    """
    The regression. With no LOGFIRE_TOKEN, importing this module used to raise
    before FastAPI was ever constructed.
    """
    import app.main

    assert app.main.app is not None


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health_reports_structure(client):
    """
    /health has to answer even when the system is unhealthy — a probe that only
    responds when everything works is not a probe.
    """
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    for field in ("status", "guardrails", "index", "mode", "auth"):
        assert field in payload, f"/health is missing {field!r}"

    assert payload["status"] in {"ok", "degraded", "unhealthy"}


def test_health_reports_which_backends_are_live(client):
    """The UI and the demo launcher both render this."""
    mode = client.get("/health").json()["mode"]
    assert set(mode) == {"vectors", "embeddings", "llm", "reranker", "guardrails"}


def test_health_reports_auth_state(client):
    """
    Deploying with API_KEY unset leaves the endpoint open to anyone who finds the
    URL. /health has to say so plainly rather than leave it ambiguous.
    """
    assert client.get("/health").json()["auth"] == "disabled"


def test_query_rejects_an_empty_question(client):
    """Pydantic min_length should reject it before any model is invoked."""
    assert client.post("/query", json={"q": ""}).status_code == 422


def test_query_rejects_an_oversized_question(client):
    assert client.post("/query", json={"q": "x" * 5000}).status_code == 422


def test_query_requires_the_question_field(client):
    assert client.post("/query", json={}).status_code == 422


def test_openapi_schema_generates(client):
    """A broken response_model shows up here rather than at request time."""
    schema = client.get("/openapi.json").json()
    assert "/query" in schema["paths"]
    assert "/health" in schema["paths"]
