"""Tests for `kagglecoach.dialogue` — the rule-based question tree."""

from __future__ import annotations

import pytest

from kagglecoach.dialogue import Dialogue, context_summary, primary_collection


# --------------------------------------------------------------------------
# Progression through the tree
# --------------------------------------------------------------------------
def test_first_question_is_task_type():
    d = Dialogue()
    q = d.next_question({})
    assert q is not None
    assert q.key == "task_type"
    assert "Tabular classification" in q.options


def test_answering_task_type_advances_to_metric():
    d = Dialogue()
    ctx = {}
    d.record_answer(ctx, "task_type", "Tabular classification")
    q = d.next_question(ctx)
    assert q is not None
    assert q.key == "metric"


def test_metric_options_branch_on_task_type_regression():
    d = Dialogue()
    ctx = {"task_type": "Tabular regression"}
    q = d.next_question(ctx)
    assert q is not None
    # Regression metrics include RMSE, MAE etc.
    assert "RMSE" in q.options
    assert "F1" not in q.options


def test_metric_options_branch_on_task_type_classification():
    d = Dialogue()
    ctx = {"task_type": "Tabular classification"}
    q = d.next_question(ctx)
    assert q is not None
    assert "ROC AUC" in q.options
    assert "RMSE" not in q.options


def test_metric_options_branch_on_task_type_nlp():
    d = Dialogue()
    ctx = {"task_type": "NLP classification"}
    q = d.next_question(ctx)
    assert q is not None
    # NLP-specific F1 variants show up
    assert any("F1" in o for o in q.options)


def test_full_walk_terminates():
    d = Dialogue()
    ctx = {}
    answers = {
        "task_type": "Tabular classification",
        "metric": "ROC AUC",
        "data_size": "Medium (10k – 100k rows)",
        "compute": "CPU only",
        "deadline": "1 – 4 weeks",
        "experience": "Some experience with similar problems",
        "goal": "Balanced — some learning, decent placement",
    }
    for key, answer in answers.items():
        q = d.next_question(ctx)
        assert q is not None
        assert q.key == key
        d.record_answer(ctx, key, answer)
    assert d.is_complete(ctx)
    assert d.next_question(ctx) is None


# --------------------------------------------------------------------------
# Validation of answers
# --------------------------------------------------------------------------
def test_invalid_option_rejected():
    d = Dialogue()
    ctx = {}
    with pytest.raises(ValueError):
        d.record_answer(ctx, "task_type", "Quantum machine learning")


def test_mismatched_key_rejected():
    d = Dialogue()
    ctx = {}
    with pytest.raises(ValueError):
        # We're on task_type, not metric.
        d.record_answer(ctx, "metric", "Log loss")


def test_recording_after_completion_raises():
    d = Dialogue()
    ctx = {
        "task_type": "Tabular classification",
        "metric": "ROC AUC",
        "data_size": "Medium (10k – 100k rows)",
        "compute": "CPU only",
        "deadline": "1 – 4 weeks",
        "experience": "Some experience with similar problems",
        "goal": "Balanced — some learning, decent placement",
    }
    with pytest.raises(RuntimeError):
        d.record_answer(ctx, "goal", "Balanced — some learning, decent placement")


# --------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------
def test_progress_counts_answered_and_total():
    d = Dialogue()
    ctx = {}
    answered, total = d.progress(ctx)
    assert answered == 0
    # Before task_type is chosen, the plan is just 1 step long.
    assert total == 1

    d.record_answer(ctx, "task_type", "Tabular classification")
    answered, total = d.progress(ctx)
    assert answered == 1
    # After task_type, the full 7-step plan is planned.
    assert total == 7


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def test_primary_collection_routes_correctly():
    assert primary_collection({"task_type": "Tabular classification"}) == "tabular"
    assert primary_collection({"task_type": "Tabular regression"}) == "tabular"
    assert primary_collection({"task_type": "NLP classification"}) == "nlp"
    assert primary_collection({"task_type": "NLP other (NER, generation, etc.)"}) == "nlp"
    assert primary_collection({"task_type": "Other / not sure"}) == "general_ml"
    assert primary_collection({}) == "general_ml"


def test_context_summary_is_readable():
    ctx = {"task_type": "Tabular classification", "metric": "ROC AUC"}
    s = context_summary(ctx)
    assert "task_type=Tabular classification" in s
    assert "metric=ROC AUC" in s


def test_context_summary_empty_case():
    s = context_summary({})
    assert "no dialogue" in s.lower()
