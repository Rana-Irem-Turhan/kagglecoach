"""
Evaluation questions bank.

Categorised prompts that exercise KaggleCoach across three axes:

    * grounded         - answerable directly from the knowledge base
    * out_of_scope     - deliberately outside KaggleCoach's domain
    * edge             - ambiguous, adversarial, or narrow

The evaluator (`run_eval.py`) scores each answer against a rubric — no
LLM judge is involved; the checks are keyword-based smoke signals so
they run in seconds with no external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    query: str
    expected_terms: Sequence[str]     # any of these terms should appear
    expected_source_contains: Sequence[str] = ()   # substring of a source filename


GROUNDED: list[EvalQuestion] = [
    EvalQuestion(
        id="grounded-lgbm-baseline",
        category="grounded",
        query="LightGBM baseline hyperparameters num_leaves min_data_in_leaf",
        expected_terms=["num_leaves", "learning_rate", "min_data_in_leaf"],
        expected_source_contains=["lightgbm"],
    ),
    EvalQuestion(
        id="grounded-catboost-cats",
        category="grounded",
        query="CatBoost categorical features ordered target statistics",
        expected_terms=["cat_features", "ordered", "target"],
        expected_source_contains=["catboost"],
    ),
    EvalQuestion(
        id="grounded-target-encoding",
        category="grounded",
        query="target encoding out of fold leakage prevention",
        expected_terms=["target encoding", "fold", "leakage"],
        expected_source_contains=["feature-engineering"],
    ),
    EvalQuestion(
        id="grounded-timeseries-validation",
        category="grounded",
        query="temporal validation TimeSeriesSplit expanding window",
        expected_terms=["TimeSeriesSplit", "temporal", "expanding"],
        expected_source_contains=["validation"],
    ),
    EvalQuestion(
        id="grounded-imbalance-strategies",
        category="grounded",
        query="handling severe class imbalance NLP strategies",
        expected_terms=["class weight", "focal loss", "threshold", "under-sampling"],
        expected_source_contains=["imbalance"],
    ),
    EvalQuestion(
        id="grounded-overfitting-audit",
        category="grounded",
        query="overfitting audit sequence feature leakage complexity",
        expected_terms=["leakage", "complexity", "regularis", "gap"],
        expected_source_contains=["overfitting"],
    ),
    EvalQuestion(
        id="grounded-metric-logloss",
        category="grounded",
        query="log loss calibration probability strategy",
        expected_terms=["calibrat", "probabilit", "log loss"],
        expected_source_contains=["metric"],
    ),
    EvalQuestion(
        id="grounded-adversarial-validation",
        category="grounded",
        query="adversarial validation train test distribution shift AUC",
        expected_terms=["adversarial", "AUC", "shift"],
        expected_source_contains=["adversarial"],
    ),
    EvalQuestion(
        id="grounded-transformer-lr",
        category="grounded",
        query="transformer fine-tuning learning rate BERT DeBERTa",
        expected_terms=["learning rate", "2e-5", "warmup"],
        expected_source_contains=["transformer"],
    ),
    EvalQuestion(
        id="grounded-tfidf-baseline",
        category="grounded",
        query="TF-IDF logistic regression ngram baseline NLP",
        expected_terms=["TF-IDF", "ngram", "Logistic"],
        expected_source_contains=["tfidf"],
    ),
]


OUT_OF_SCOPE: list[EvalQuestion] = [
    EvalQuestion(
        id="oos-recipe",
        category="out_of_scope",
        query="How do I make a good chocolate mousse from scratch?",
        expected_terms=[],   # we don't expect useful terms; we want the confidence gate to trip
    ),
    EvalQuestion(
        id="oos-world-history",
        category="out_of_scope",
        query="Who was the Roman emperor in 79 AD?",
        expected_terms=[],
    ),
    EvalQuestion(
        id="oos-programming-general",
        category="out_of_scope",
        query="Explain the visitor pattern in object-oriented programming",
        expected_terms=[],
    ),
]


EDGE: list[EvalQuestion] = [
    EvalQuestion(
        id="edge-vague-model",
        category="edge",
        query="which model should I use",
        expected_terms=["LightGBM", "gradient boosting", "TF-IDF"],  # any signal is fine
    ),
    EvalQuestion(
        id="edge-single-word",
        category="edge",
        query="overfitting",
        expected_terms=["gap", "regularis", "leakage"],
    ),
    EvalQuestion(
        id="edge-off-domain-adjacent",
        category="edge",
        query="how to preprocess images for a CNN",
        expected_terms=[],   # KaggleCoach's KB doesn't cover images
    ),
]


ALL_QUESTIONS: list[EvalQuestion] = GROUNDED + OUT_OF_SCOPE + EDGE
