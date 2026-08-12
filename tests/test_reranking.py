"""
Reranking contract.

The reranker is the stage most likely to fail in production — it loads an ONNX
model from a cache directory that may not be writable. Its documented promise is
that failure degrades answer quality but never costs the user their answer, and
that source metadata survives the ranking step. Both are tested here without
loading the real model.
"""

import pytest

from app.services.retrieval import ranking_service


def _docs(n=4):
    return [
        {"content": f"chunk {i} about kubernetes", "source": f"doc_{i}.pdf", "score": 0.9 - i * 0.1}
        for i in range(n)
    ]


def test_empty_input_returns_empty():
    assert ranking_service.rerank_documents("query", []) == []


def test_source_metadata_survives_reranking(monkeypatch):
    """
    The regression this file exists for. The retriever used to reduce chunks to
    bare content strings, which made source attribution impossible downstream —
    the API could return text but never say where it came from.
    """
    class FakeRanker:
        def rerank(self, request):
            # Reverse the input order so the mapping back to originals is
            # actually exercised rather than accidentally correct.
            return [
                {"id": i, "text": p["text"], "score": 0.5 + i * 0.1}
                for i, p in reversed(list(enumerate(request.passages)))
            ]

    monkeypatch.setattr(ranking_service, "_get_ranker", lambda: FakeRanker())

    result = ranking_service.rerank_documents("kubernetes", _docs(4), top_n=3)

    assert len(result) == 3
    for chunk in result:
        assert "source" in chunk and chunk["source"].endswith(".pdf")
        assert "content" in chunk
        assert "rerank_score" in chunk


def test_content_stays_paired_with_its_own_source(monkeypatch):
    """A mismatched id→document mapping would cite the wrong file — silently."""
    class ReversingRanker:
        def rerank(self, request):
            return [
                {"id": i, "text": p["text"], "score": float(i)}
                for i, p in reversed(list(enumerate(request.passages)))
            ]

    monkeypatch.setattr(ranking_service, "_get_ranker", lambda: ReversingRanker())

    for chunk in ranking_service.rerank_documents("q", _docs(4), top_n=4):
        index = chunk["content"].split()[1]  # "chunk 2 about ..." -> "2"
        assert chunk["source"] == f"doc_{index}.pdf"


def test_ranker_failure_degrades_instead_of_raising(monkeypatch):
    """A dead reranker costs quality. It must not cost the user their answer."""
    def explode():
        raise RuntimeError("ONNX model could not be loaded")

    monkeypatch.setattr(ranking_service, "_get_ranker", explode)

    docs = _docs(6)
    result = ranking_service.rerank_documents("query", docs, top_n=3)

    assert result == docs[:3]  # falls back to Qdrant's cosine ordering


def test_top_n_is_respected(monkeypatch):
    class PassthroughRanker:
        def rerank(self, request):
            return [{"id": i, "text": p["text"], "score": 1.0} for i, p in enumerate(request.passages)]

    monkeypatch.setattr(ranking_service, "_get_ranker", lambda: PassthroughRanker())
    assert len(ranking_service.rerank_documents("q", _docs(10), top_n=5)) == 5
