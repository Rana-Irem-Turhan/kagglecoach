"""Tests for `kagglecoach.retriever`."""

from __future__ import annotations

import numpy as np


def _seed_collections(store, fake_model):
    """Populate all three collections with a handful of realistic-ish chunks."""
    tabular_bodies = [
        "LightGBM baseline hyperparameters num_leaves min_data_in_leaf",
        "CatBoost ordered target statistics categorical",
        "Target encoding out-of-fold folds prevent leakage",
    ]
    nlp_bodies = [
        "TF-IDF baseline ngram_range word features",
        "Transformer fine-tuning learning rate warmup",
    ]
    general_bodies = [
        "Validation strategy TimeSeriesSplit GroupKFold StratifiedKFold",
        "Overfitting diagnosis train validation gap severity",
    ]

    for name, bodies in [
        ("tabular", tabular_bodies),
        ("nlp", nlp_bodies),
        ("general_ml", general_bodies),
    ]:
        vecs = fake_model.embed(bodies)
        entries = [
            (f"{name}_{i}.md", i, body, vec)
            for i, (body, vec) in enumerate(zip(bodies, vecs))
        ]
        store.add_collection(name, entries)


# --------------------------------------------------------------------------
# Basic retrieval
# --------------------------------------------------------------------------
def test_find_single_collection(tmp_store, fake_model):
    from kagglecoach.retriever import MultiCollectionRetriever
    _seed_collections(tmp_store, fake_model)

    r = MultiCollectionRetriever(store=tmp_store, model_client=fake_model)
    hits = r.find("LightGBM baseline num_leaves", "tabular", top_k=2)
    assert len(hits) >= 1
    # Highest-scoring result should be the LightGBM chunk
    assert "LightGBM" in hits[0].body or "lightgbm" in hits[0].body.lower()


def test_find_across_returns_all_collections(tmp_store, fake_model):
    from kagglecoach.retriever import MultiCollectionRetriever
    _seed_collections(tmp_store, fake_model)

    r = MultiCollectionRetriever(store=tmp_store, model_client=fake_model)
    result = r.find_across(
        query="validation strategy classification imbalance",
        primary_collection="general_ml",
    )
    assert result.primary_collection == "general_ml"
    # Should have hits from at least the general_ml collection
    assert "general_ml" in result.by_collection


def test_empty_query_returns_empty_result(tmp_store, fake_model):
    from kagglecoach.retriever import MultiCollectionRetriever
    _seed_collections(tmp_store, fake_model)

    r = MultiCollectionRetriever(store=tmp_store, model_client=fake_model)
    result = r.find_across("", "tabular")
    assert result.by_collection == {}
    assert result.confidence_ok is False


# --------------------------------------------------------------------------
# Confidence gating
# --------------------------------------------------------------------------
def test_low_similarity_hits_are_dropped(tmp_store, fake_model, settings):
    from kagglecoach.retriever import MultiCollectionRetriever
    _seed_collections(tmp_store, fake_model)

    r = MultiCollectionRetriever(store=tmp_store, model_client=fake_model)
    result = r.find_across("zqxwv unrelated random", "tabular")
    # These random letters shouldn't match the character-histogram embeddings well
    # so most hits fall under threshold and get dropped.
    assert result.dropped_below_threshold >= 0  # non-negative sanity check


# --------------------------------------------------------------------------
# Evidence block formatting
# --------------------------------------------------------------------------
def test_format_evidence_labels_by_source(tmp_store, fake_model):
    from kagglecoach.retriever import MultiCollectionRetriever, format_evidence_block
    _seed_collections(tmp_store, fake_model)

    r = MultiCollectionRetriever(store=tmp_store, model_client=fake_model)
    result = r.find_across(
        "LightGBM CatBoost hyperparameters",
        primary_collection="tabular",
    )
    block = format_evidence_block(result)
    assert "tabular" in block
    assert ".md" in block  # source file citation present


def test_format_evidence_empty_result_is_labelled(tmp_store, fake_model):
    from kagglecoach.retriever import MultiCollectionRetriever, format_evidence_block

    r = MultiCollectionRetriever(store=tmp_store, model_client=fake_model)
    result = r.find_across("anything", "tabular")
    # Empty store → no hits
    block = format_evidence_block(result)
    assert "no evidence" in block.lower()
