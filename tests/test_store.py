"""Tests for `kagglecoach.store` (FAISS + SQLite)."""

from __future__ import annotations

import numpy as np
import pytest


def _make_entries(prefix: str, count: int, dim: int = 32):
    """Fake (source_file, section_idx, body, vector) tuples."""
    rng = np.random.default_rng(seed=42)
    entries = []
    for i in range(count):
        body = f"{prefix} chunk {i} " + "content " * (10 + i)
        vec = rng.random(dim).astype(np.float32)
        entries.append((f"{prefix}.md", i, body, vec))
    return entries


def test_empty_store_reports_zero(tmp_store):
    assert tmp_store.size() == 0
    assert tmp_store.collection_names() == []
    assert tmp_store.source_files() == []


def test_add_and_search_single_collection(tmp_store):
    entries = _make_entries("lightgbm", count=6)
    added = tmp_store.add_collection("tabular", entries)
    assert added == 6
    assert tmp_store.size() == 6
    assert tmp_store.size("tabular") == 6

    # Search with the first entry's own vector — it should come back top.
    hits = tmp_store.search("tabular", entries[0][3], top_k=3)
    assert len(hits) == 3
    assert hits[0][0].body == entries[0][2]
    # Cosine similarity with the same vector is ~1.
    assert hits[0][1] > 0.99


def test_search_orders_by_similarity(tmp_store):
    entries = _make_entries("test", count=10, dim=16)
    tmp_store.add_collection("tabular", entries)
    query = entries[3][3]  # target: entry index 3
    hits = tmp_store.search("tabular", query, top_k=5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    # The exact match should be at the top.
    assert hits[0][0].section_idx == 3


def test_missing_collection_returns_empty(tmp_store):
    entries = _make_entries("nlp", count=3)
    tmp_store.add_collection("nlp", entries)
    query = entries[0][3]
    assert tmp_store.search("does-not-exist", query, top_k=3) == []


def test_multiple_collections_are_independent(tmp_store):
    e1 = _make_entries("tab", count=4, dim=32)
    e2 = _make_entries("nlp", count=5, dim=32)
    tmp_store.add_collection("tabular", e1)
    tmp_store.add_collection("nlp", e2)

    assert tmp_store.size() == 9
    assert tmp_store.size("tabular") == 4
    assert tmp_store.size("nlp") == 5
    assert set(tmp_store.collection_names()) == {"tabular", "nlp"}


def test_wipe_collection_leaves_others_intact(tmp_store):
    e1 = _make_entries("tab", count=3, dim=16)
    e2 = _make_entries("nlp", count=4, dim=16)
    tmp_store.add_collection("tabular", e1)
    tmp_store.add_collection("nlp", e2)

    tmp_store.wipe_collection("tabular")
    assert tmp_store.size("tabular") == 0
    assert tmp_store.size("nlp") == 4


def test_wipe_all(tmp_store):
    e1 = _make_entries("x", count=3, dim=16)
    tmp_store.add_collection("tabular", e1)
    tmp_store.wipe()
    assert tmp_store.size() == 0


def test_stats_are_accurate(tmp_store):
    entries = _make_entries("stats", count=5, dim=24)
    tmp_store.add_collection("tabular", entries)
    stats = tmp_store.stats("tabular")
    assert stats.name == "tabular"
    assert stats.chunk_count == 5
    assert stats.file_count == 1
    assert stats.embedding_dim == 24
    # Every body length should be positive.
    assert stats.avg_chunk_chars > 0


def test_zero_query_returns_empty(tmp_store):
    entries = _make_entries("x", count=3, dim=16)
    tmp_store.add_collection("tabular", entries)
    hits = tmp_store.search("tabular", np.zeros(16, dtype=np.float32), top_k=3)
    assert hits == []


def test_persistence_across_reopen(tmp_path):
    from kagglecoach.store import MultiCollectionStore
    entries = _make_entries("keep", count=4, dim=16)
    with MultiCollectionStore(tmp_path / "p.sqlite", tmp_path / "faiss") as s1:
        s1.add_collection("tabular", entries)

    with MultiCollectionStore(tmp_path / "p.sqlite", tmp_path / "faiss") as s2:
        assert s2.size("tabular") == 4
        hits = s2.search("tabular", entries[0][3], top_k=1)
        assert len(hits) == 1
        assert hits[0][0].body == entries[0][2]
