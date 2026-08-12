"""
Shared test setup.

These are unit tests: no network, no API keys, no Qdrant, no LLM. Every module
under test imports logfire and app.config at load time though, so a couple of
things need to be neutralised before collection or importing anything raises.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Placeholder credentials. Client objects are constructed at import time in
# several modules; they never make a call during these tests, but they do refuse
# to be built from None.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("GROQ_FALLBACK_API_KEY", "test-key")
os.environ.setdefault("PORTKEY_API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("QDRANT_API_KEY", "test-key")
os.environ.setdefault("QDRANT_CLUSTER_ENDPOINT", "http://localhost:6333")

# Keep traces and LangSmith runs out of the collector during test runs.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import logfire

logfire.configure(send_to_logfire=False, console=False)
