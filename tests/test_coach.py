"""End-to-end coach test using the fake model client."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _seed_all_collections(store, fake_model):
    entries = {
        "tabular": [
            "LightGBM baseline num_leaves=63 learning_rate=0.05",
            "CatBoost ordered target statistics categorical features",
            "Target encoding out-of-fold prevents leakage",
            "Feature engineering datetime decomposition year month day",
        ],
        "nlp": [
            "TF-IDF ngram_range word features baseline",
            "Transformer fine-tuning learning rate 2e-5",
        ],
        "general_ml": [
            "TimeSeriesSplit temporal validation expanding window",
            "StratifiedKFold classification imbalance preserves class ratios",
            "Overfitting train validation gap severity classification",
        ],
    }
    for name, bodies in entries.items():
        vecs = fake_model.embed(bodies)
        store.add_collection(
            name,
            [(f"{name}_{i}.md", i, body, vec)
             for i, (body, vec) in enumerate(zip(bodies, vecs))],
        )


# --------------------------------------------------------------------------
# Full run_session flow
# --------------------------------------------------------------------------
def test_full_session_produces_report(tmp_store, fake_model):
    from kagglecoach.coach import Coach

    _seed_all_collections(tmp_store, fake_model)
    coach = Coach(store=tmp_store, model_client=fake_model)

    ctx = {
        "task_type": "Tabular classification",
        "metric": "ROC AUC",
        "data_size": "Medium (10k – 100k rows)",
        "compute": "CPU only",
        "deadline": "1 – 4 weeks",
        "experience": "Some experience with similar problems",
        "goal": "Balanced — some learning, decent placement",
    }

    df = pd.DataFrame({
        "x1": np.random.rand(1000),
        "x2": np.random.rand(1000),
        "cat": np.random.choice(["A", "B", "C"], 1000),
        "target": np.random.randint(0, 2, 1000),
    })

    report = coach.run_session(
        dialogue_context=ctx,
        uploaded_df=df,
        target_column="target",
    )

    # Rule-based sections always populated
    assert report.dataset_profile is not None
    assert report.dataset_profile.task_type == "binary"
    assert report.overfitting is not None
    assert report.validation is not None
    assert report.primary_collection == "tabular"

    # LLM sections should be non-empty strings (from FakeModel echo)
    assert report.model_selection_prose
    assert report.feature_engineering_prose

    # Retrieval populated
    assert report.retrieval.by_collection


def test_session_without_dataset_still_works(tmp_store, fake_model):
    from kagglecoach.coach import Coach
    _seed_all_collections(tmp_store, fake_model)
    coach = Coach(store=tmp_store, model_client=fake_model)

    ctx = {
        "task_type": "NLP classification",
        "metric": "F1 (macro)",
        "data_size": "Small (< 10k rows)",
        "compute": "CPU only",
        "deadline": "Less than a week",
        "experience": "First competition / new to this area",
        "goal": "Learning a specific technique",
    }
    report = coach.run_session(dialogue_context=ctx)  # no dataset

    assert report.dataset_profile is None
    assert report.primary_collection == "nlp"
    # Overfitting still gets an assessment from dialogue-only estimation
    assert report.overfitting is not None


def test_report_flags_low_confidence_when_no_evidence(tmp_store, fake_model):
    """Empty knowledge base → confidence_ok false → warnings raised."""
    from kagglecoach.coach import Coach
    # Do NOT seed collections — store is empty.
    coach = Coach(store=tmp_store, model_client=fake_model)

    ctx = {
        "task_type": "Tabular classification",
        "metric": "ROC AUC",
        "data_size": "Medium (10k – 100k rows)",
        "compute": "CPU only",
        "deadline": "1 – 4 weeks",
        "experience": "Some experience with similar problems",
        "goal": "Balanced — some learning, decent placement",
    }
    report = coach.run_session(dialogue_context=ctx)
    assert not report.retrieval.confidence_ok
    assert any("strong hits" in w.lower() or "less grounded" in w.lower()
               for w in report.warnings)


def test_generation_mode_reflects_client_state(tmp_store, fake_model):
    from kagglecoach.coach import Coach
    _seed_all_collections(tmp_store, fake_model)
    coach = Coach(store=tmp_store, model_client=fake_model)

    ctx = {
        "task_type": "Tabular classification",
        "metric": "ROC AUC",
        "data_size": "Medium (10k – 100k rows)",
        "compute": "CPU only",
        "deadline": "1 – 4 weeks",
        "experience": "Some experience with similar problems",
        "goal": "Balanced — some learning, decent placement",
    }
    report = coach.run_session(dialogue_context=ctx)
    assert report.generation_mode == "local"

    fake_model.use_azure_for_chat = True
    report2 = coach.run_session(dialogue_context=ctx)
    assert report2.generation_mode == "azure"


# --------------------------------------------------------------------------
# Report renderer
# --------------------------------------------------------------------------
def test_report_renders_to_markdown(tmp_store, fake_model):
    from kagglecoach.coach import Coach
    from kagglecoach.report import render_report

    _seed_all_collections(tmp_store, fake_model)
    coach = Coach(store=tmp_store, model_client=fake_model)

    ctx = {
        "task_type": "Tabular classification",
        "metric": "ROC AUC",
        "data_size": "Medium (10k – 100k rows)",
        "compute": "CPU only",
        "deadline": "1 – 4 weeks",
        "experience": "Some experience with similar problems",
        "goal": "Balanced — some learning, decent placement",
    }
    report = coach.run_session(dialogue_context=ctx)
    md = render_report(report)

    # Structural sanity checks
    assert md.startswith("# ")
    assert "## 1. User context" in md
    assert "## 3. Recommended pipeline" in md
    assert "## 6. Overfitting assessment" in md
    assert "## 7. Validation strategy" in md
    assert "## 8. Retrieved evidence" in md
