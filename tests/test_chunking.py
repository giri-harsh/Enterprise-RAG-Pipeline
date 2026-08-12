"""
Tests for the paragraph chunker.

The chunker is the least glamorous part of the pipeline and the one most likely
to quietly break retrieval. If it emits empty strings, the embedding API rejects
the batch. If it silently drops the tail paragraph, documents lose their
conclusions. Neither failure raises — they just make answers worse.
"""

import pytest

from app.ingestion.chunking.splitter import chunk_text


def test_empty_input_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  \t ") == []


def test_short_text_stays_one_chunk():
    chunks = chunk_text("A single short paragraph about Kubernetes pods.")
    assert len(chunks) == 1
    assert "Kubernetes pods" in chunks[0]


def test_no_chunk_is_blank():
    """Blank chunks are rejected by the embedding API — they must never be emitted."""
    text = "\n\n".join(["Real paragraph.", "", "   ", "Another real paragraph.", ""])
    assert all(c.strip() for c in chunk_text(text))


def test_long_text_splits_into_multiple_chunks():
    paragraph = "Kubernetes schedules pods onto nodes. " * 20  # ~740 chars
    chunks = chunk_text("\n\n".join([paragraph] * 6), chunk_size=1500)
    assert len(chunks) > 1


def test_final_paragraph_is_never_dropped():
    """Regression guard: the tail buffer must be flushed after the loop ends."""
    marker = "UNIQUE_TAIL_MARKER_9137"
    text = "\n\n".join(["filler paragraph " * 60] * 4 + [marker])
    assert any(marker in c for c in chunk_text(text, chunk_size=800))


def test_all_content_is_preserved():
    """Chunking rearranges whitespace but must not lose words."""
    text = "\n\n".join(f"Paragraph {i} discusses SRIOV configuration." for i in range(12))
    joined = " ".join(chunk_text(text, chunk_size=400))
    for i in range(12):
        assert f"Paragraph {i}" in joined


def test_oversized_single_paragraph_is_not_lost():
    """
    A paragraph longer than chunk_size cannot be split by this chunker — it packs
    paragraphs, it does not break them. The documented behaviour is that such a
    paragraph is emitted whole and exceeds the limit. This test pins that down so
    the tradeoff is a known one rather than a surprise at ingestion time.
    """
    giant = "x" * 5000
    chunks = chunk_text(giant, chunk_size=1500)
    assert len(chunks) == 1
    assert len(chunks[0]) == 5000
