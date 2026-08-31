"""
Coach orchestrator.

Ties dialogue + EDA + overfitting + validation + retrieval + LLM into a
single `run_session()` call that returns a `StrategyReport` — the
structured object rendered by `report.py`.

Flow (deterministic → LLM → deterministic):

    1. Rule-based EDA on the uploaded CSV (if any).
    2. Rule-based overfitting risk assessment from dataset shape and
       user-supplied scores.
    3. Rule-based validation strategy selection.
    4. Formulate two RAG queries: model selection + feature engineering.
    5. Retrieve grounded evidence from the three FAISS collections.
    6. LLM synthesises the natural-language sections using the retrieved
       evidence — this is the only step that touches the model. Numbers
       and strategy calls come from the rule-based layers above.

Public API:
    coach = Coach(store, model_client)
    report = coach.run_session(dialogue_context, uploaded_df=None,
                               train_score=None, val_score=None)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from kagglecoach.dialogue import context_summary, primary_collection
from kagglecoach.eda import DatasetProfile, profile_dataframe
from kagglecoach.models import ModelClient
from kagglecoach.overfitting import RiskAssessment, assess_overfitting
from kagglecoach.retriever import (
    MultiCollectionRetriever,
    RetrievalResult,
    format_evidence_block,
)
from kagglecoach.settings import SETTINGS, Settings
from kagglecoach.store import MultiCollectionStore
from kagglecoach.validation import (
    ValidationRecommendation,
    recommend_validation,
)


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------
@dataclass
class StrategyReport:
    """The full report handed off to the renderer."""

    # Context ---------------------------------------------------------
    dialogue_summary: str
    dataset_profile: Optional[DatasetProfile]

    # Rule-based sections --------------------------------------------
    overfitting: RiskAssessment
    validation: ValidationRecommendation
    primary_collection: str

    # LLM-generated sections ------------------------------------------
    model_selection_prose: str
    feature_engineering_prose: str

    # Evidence + metadata --------------------------------------------
    retrieval: RetrievalResult
    generation_mode: str                              # 'local' or 'azure'
    warnings: list[str] = field(default_factory=list)

    @property
    def evidence_summary(self) -> dict[str, list[str]]:
        """Filename list per collection — for the report citations block."""
        out: dict[str, list[str]] = {}
        for coll, matches in self.retrieval.by_collection.items():
            seen: list[str] = []
            for m in matches:
                if m.source_file not in seen:
                    seen.append(m.source_file)
            out[coll] = seen
        return out


# --------------------------------------------------------------------------
# Coach
# --------------------------------------------------------------------------
@dataclass
class Coach:
    store: MultiCollectionStore
    model_client: ModelClient
    settings: Settings = field(default_factory=lambda: SETTINGS)

    def __post_init__(self) -> None:
        self._retriever = MultiCollectionRetriever(
            store=self.store,
            model_client=self.model_client,
            settings=self.settings,
        )

    # -- main entry ------------------------------------------------------
    def run_session(
        self,
        dialogue_context: dict[str, str],
        uploaded_df: Optional[pd.DataFrame] = None,
        target_column: Optional[str] = None,
        train_score: Optional[float] = None,
        val_score: Optional[float] = None,
        higher_is_better: bool = True,
    ) -> StrategyReport:
        """Build a full StrategyReport for the given user context."""
        warnings: list[str] = []
        primary = primary_collection(dialogue_context)

        # 1) EDA ---------------------------------------------------------
        profile: Optional[DatasetProfile] = None
        if uploaded_df is not None:
            try:
                profile = profile_dataframe(
                    uploaded_df,
                    target_column=target_column,
                    settings=self.settings,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"EDA failed: {exc}")

        # 2) Overfitting -------------------------------------------------
        if profile is not None:
            n_rows = profile.n_rows
            # Subtract target column if present when counting features.
            n_features = profile.n_columns - (1 if target_column else 0)
        else:
            n_rows = _size_bucket_to_row_estimate(dialogue_context.get("data_size", ""))
            n_features = 20  # generic assumption when no dataset is uploaded

        overfitting = assess_overfitting(
            n_rows=n_rows,
            n_features=max(n_features, 1),
            train_score=train_score,
            val_score=val_score,
            higher_is_better=higher_is_better,
            settings=self.settings,
        )

        # 3) Validation --------------------------------------------------
        validation = recommend_validation(
            profile=profile,
            dialogue_context=dialogue_context,
            train_score_available=train_score is not None,
        )

        # 4) Retrieval ---------------------------------------------------
        model_query = self._build_model_query(dialogue_context, profile)
        feature_query = self._build_feature_query(dialogue_context, profile)

        retrieval_model = self._retriever.find_across(model_query, primary)
        retrieval_feature = self._retriever.find_across(feature_query, primary)
        retrieval_combined = _combine_retrievals(retrieval_model, retrieval_feature)

        if not retrieval_combined.confidence_ok:
            warnings.append(
                f"Fewer than {self.settings.min_strong_hits_primary} strong hits in the "
                f"primary collection ({primary}) — recommendations are less grounded than usual."
            )

        # 5) LLM prose sections ------------------------------------------
        model_prose = self._llm_model_section(
            dialogue_context, profile, retrieval_model,
        )
        feature_prose = self._llm_feature_section(
            dialogue_context, profile, retrieval_feature,
        )

        return StrategyReport(
            dialogue_summary=context_summary(dialogue_context),
            dataset_profile=profile,
            overfitting=overfitting,
            validation=validation,
            primary_collection=primary,
            model_selection_prose=model_prose,
            feature_engineering_prose=feature_prose,
            retrieval=retrieval_combined,
            generation_mode=self.model_client.active_mode,
            warnings=warnings,
        )

    # -- LLM section prompts --------------------------------------------
    def _llm_model_section(
        self,
        context: dict[str, str],
        profile: Optional[DatasetProfile],
        retrieval: RetrievalResult,
    ) -> str:
        if not retrieval.by_collection:
            return ("Insufficient evidence retrieved from the knowledge base to "
                    "generate a grounded model-selection recommendation.")
        prompt = self._compose_prompt(
            task="Recommend which model families to try first for this competition, "
                 "and give a concrete baseline configuration (one primary, one alternative). "
                 "Explain trade-offs given the user's constraints. "
                 "Do NOT invent hyperparameter values — only reference values that appear "
                 "in the EVIDENCE. Cite source filenames.",
            context=context,
            profile=profile,
            retrieval=retrieval,
        )
        return self.model_client.chat(self.settings.system_prompt, prompt)

    def _llm_feature_section(
        self,
        context: dict[str, str],
        profile: Optional[DatasetProfile],
        retrieval: RetrievalResult,
    ) -> str:
        if not retrieval.by_collection:
            return ("Insufficient evidence retrieved from the knowledge base to "
                    "generate a grounded feature-engineering plan.")
        prompt = self._compose_prompt(
            task="Suggest 3-5 concrete feature-engineering steps for this dataset, "
                 "prioritised by expected impact. Warn about any specific leakage risks "
                 "the dataset profile implies. Cite source filenames.",
            context=context,
            profile=profile,
            retrieval=retrieval,
        )
        return self.model_client.chat(self.settings.system_prompt, prompt)

    def _compose_prompt(
        self,
        task: str,
        context: dict[str, str],
        profile: Optional[DatasetProfile],
        retrieval: RetrievalResult,
    ) -> str:
        parts: list[str] = []
        parts.append(f"TASK: {task}")
        parts.append("")
        parts.append("USER CONTEXT:")
        parts.append(context_summary(context))
        parts.append("")
        if profile is not None:
            parts.append("DATASET PROFILE:")
            parts.extend(f"- {b}" for b in profile.summary_bullets)
            parts.append("")
        parts.append("EVIDENCE:")
        parts.append(format_evidence_block(retrieval))
        parts.append("")
        parts.append("Answer:")
        return "\n".join(parts)

    # -- retrieval query construction ----------------------------------
    def _build_model_query(
        self,
        context: dict[str, str],
        profile: Optional[DatasetProfile],
    ) -> str:
        task = context.get("task_type", "").lower()
        metric = context.get("metric", "")
        size = context.get("data_size", "")

        parts = ["model selection strategy hyperparameters"]
        if "tabular" in task:
            parts.append("gradient boosting")
        if "nlp" in task:
            parts.append("transformer TF-IDF baseline")
        if metric:
            parts.append(f"for {metric}")
        if size:
            parts.append(size)
        if profile is not None:
            parts.append(f"{profile.size_bucket} dataset")
            if profile.task_type in ("binary", "multiclass"):
                parts.append(f"{profile.task_type} classification")
            elif profile.task_type == "regression":
                parts.append("regression")
        return " ".join(parts)

    def _build_feature_query(
        self,
        context: dict[str, str],
        profile: Optional[DatasetProfile],
    ) -> str:
        task = context.get("task_type", "").lower()
        parts = ["feature engineering encoding leakage"]
        if "tabular" in task:
            parts.append("categorical target encoding datetime aggregation")
        if "nlp" in task:
            parts.append("text preprocessing ngram tokenisation")
        if profile is not None:
            if profile.high_cardinality_categorical:
                parts.append("high cardinality categorical target encoding")
            if profile.datetime_columns:
                parts.append("datetime decomposition temporal features")
            if profile.text_columns:
                parts.append("text feature")
            if profile.imbalance_severity in ("moderate", "severe"):
                parts.append("class imbalance")
        return " ".join(parts)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _size_bucket_to_row_estimate(size_answer: str) -> int:
    """Map the dialogue's size answer to a representative row count."""
    s = size_answer.lower()
    if "very large" in s:
        return 5_000_000
    if "large" in s:
        return 500_000
    if "medium" in s:
        return 50_000
    if "small" in s:
        return 5_000
    return 10_000  # unknown → moderate default


def _combine_retrievals(*results: RetrievalResult) -> RetrievalResult:
    """Merge multiple RetrievalResults for reporting purposes."""
    combined_by_coll: dict[str, list[object]] = {}
    seen_ids: set[tuple[str, int]] = set()

    dropped_total = 0
    primary = results[0].primary_collection if results else ""
    primary_strong = 0
    confidence_ok = False

    for r in results:
        dropped_total += r.dropped_below_threshold
        primary_strong = max(primary_strong, r.primary_strong_hits)
        confidence_ok = confidence_ok or r.confidence_ok
        for coll, matches in r.by_collection.items():
            bucket = combined_by_coll.setdefault(coll, [])
            for m in matches:
                key = (m.chunk.source_file, m.chunk.section_idx)
                if key not in seen_ids:
                    seen_ids.add(key)
                    bucket.append(m)

    # Sort each collection's list by descending score.
    for coll, matches in combined_by_coll.items():
        matches.sort(key=lambda m: m.score, reverse=True)  # type: ignore[arg-type]

    return RetrievalResult(
        by_collection=combined_by_coll,  # type: ignore[arg-type]
        primary_collection=primary,
        primary_strong_hits=primary_strong,
        confidence_ok=confidence_ok,
        dropped_below_threshold=dropped_total,
    )
