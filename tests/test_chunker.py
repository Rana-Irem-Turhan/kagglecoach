"""Tests for `kagglecoach.chunker`."""

from __future__ import annotations

import pytest

from kagglecoach.chunker import chunk_by_sections


# --------------------------------------------------------------------------
# Header splitting
# --------------------------------------------------------------------------
def test_splits_at_h2_headers():
    text = (
        "# Title\n\n"
        + ("Intro paragraph. " * 20)
        + "\n\n## Section A\n\n"
        + ("Body A. " * 40)
        + "\n\n## Section B\n\n"
        + ("Body B. " * 40)
    )
    chunks = chunk_by_sections(text, min_chars=100, max_chars=1000)
    assert len(chunks) >= 2
    # Every chunk after the first should begin at a heading marker.
    section_chunks = [c for c in chunks if c.lstrip().startswith("##")]
    assert len(section_chunks) >= 2


def test_h3_headers_also_split():
    text = (
        "## Parent\n\n" + ("Parent body. " * 20)
        + "\n\n### Child A\n\n" + ("Child A body. " * 30)
        + "\n\n### Child B\n\n" + ("Child B body. " * 30)
    )
    chunks = chunk_by_sections(text, min_chars=100, max_chars=800)
    assert any(c.lstrip().startswith("### Child A") for c in chunks)
    assert any(c.lstrip().startswith("### Child B") for c in chunks)


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------
def test_respects_max_chars():
    text = "## Section\n\n" + ("Sentence. " * 500)
    chunks = chunk_by_sections(text, min_chars=200, max_chars=800)
    assert all(len(c) <= 800 for c in chunks), \
        f"Chunk sizes: {[len(c) for c in chunks]}"


def test_undersized_trailing_merges_upward():
    text = (
        "## Main\n\n" + ("Detailed body. " * 40)
        + "\n\n## Tiny\n\nshort"
    )
    chunks = chunk_by_sections(text, min_chars=200, max_chars=2000)
    assert all(len(c) >= 200 or c is chunks[-1] for c in chunks)
    joined = "\n".join(chunks)
    assert "Tiny" in joined and "short" in joined


def test_undersized_leading_title_prepends():
    text = "## Tiny title\n\n" + ("Substantive body. " * 60)
    chunks = chunk_by_sections(text, min_chars=200, max_chars=2000)
    assert len(chunks) == 1
    assert "Tiny title" in chunks[0]


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------
def test_empty_input_returns_empty():
    assert chunk_by_sections("", 100, 500) == []
    assert chunk_by_sections("   \n\n  \n", 100, 500) == []


def test_no_headers_falls_back_to_paragraphs():
    text = "\n\n".join(["Paragraph " + "x" * 200] * 5)
    chunks = chunk_by_sections(text, min_chars=100, max_chars=600)
    assert len(chunks) >= 2
    assert all(len(c) <= 600 for c in chunks)


def test_oversized_paragraph_is_hard_sliced():
    text = "## Big\n\n" + "x" * 5000
    chunks = chunk_by_sections(text, min_chars=200, max_chars=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(len(c) for c in chunks) >= 5000  # nothing lost


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        chunk_by_sections("text", 500, 100)  # min > max
    with pytest.raises(ValueError):
        chunk_by_sections("text", 100, 0)    # max <= 0


# --------------------------------------------------------------------------
# Real knowledge base file
# --------------------------------------------------------------------------
def test_real_lightgbm_doc_chunks_reasonably():
    from pathlib import Path
    text = (Path(__file__).parent.parent
            / "knowledge" / "tabular" / "01-lightgbm-strategy.md").read_text()
    chunks = chunk_by_sections(text, min_chars=300, max_chars=1600)
    assert len(chunks) >= 3
    assert all(300 <= len(c) <= 1600 for c in chunks), \
        f"Real doc chunk sizes: {[len(c) for c in chunks]}"
