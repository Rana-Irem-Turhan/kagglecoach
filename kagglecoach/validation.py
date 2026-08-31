"""
Rule-based validation strategy selector.

Given the dataset profile plus the user's dialogue context, recommend a
specific validation strategy and explain why. This is deterministic —
the decision tree lives here in Python code, not in an LLM prompt.

The strategy names align with knowledge base sections so retrieval can
fetch the corresponding explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from kagglecoach.eda import DatasetProfile


@dataclass
class ValidationRecommendation:
    strategy: str                     # e.g. 'TimeSeriesSplit', 'StratifiedKFold'
    n_splits: int
    reason: str                       # short human-readable rationale
    warnings: list[str] = field(default_factory=list)
    fallback: Optional[str] = None    # alternative strategy if primary is impractical


def recommend_validation(
    profile: Optional[DatasetProfile],
    dialogue_context: dict[str, str],
    train_score_available: bool = False,
) -> ValidationRecommendation:
    """
    Walk the decision tree top-to-bottom.

    Order of checks — first match wins:
        1. Temporal structure → TimeSeriesSplit.
        2. Group structure → GroupKFold or StratifiedGroupKFold.
        3. Classification imbalance → StratifiedKFold.
        4. Otherwise → KFold with shuffle.
    """
    warnings: list[str] = []

    # No profile — fall back to dialogue-only reasoning.
    if profile is None:
        return _from_dialogue_only(dialogue_context)

    # 1) Temporal --------------------------------------------------------
    if profile.datetime_columns:
        return ValidationRecommendation(
            strategy="TimeSeriesSplit",
            n_splits=5,
            reason=(
                f"Datetime column(s) detected ({', '.join(profile.datetime_columns)}). "
                "Random splits would let the model see the future during training. "
                "TimeSeriesSplit produces expanding-window folds that respect temporal order."
            ),
            warnings=[
                "Every rolling feature, target encoding, and aggregation must use only "
                "past rows within its fold — otherwise validation still leaks the future."
            ],
        )

    # 2) Group -----------------------------------------------------------
    if profile.likely_group_columns:
        group_cols = ", ".join(profile.likely_group_columns[:3])
        if profile.task_type in ("binary", "multiclass"):
            strategy = "StratifiedGroupKFold"
            reason_extra = (
                "Task is classification, so we stratify within group folds "
                "to preserve class ratios."
            )
        else:
            strategy = "GroupKFold"
            reason_extra = "Non-classification task — plain GroupKFold is sufficient."
        return ValidationRecommendation(
            strategy=strategy,
            n_splits=5,
            reason=(
                f"Likely group column(s) detected ({group_cols}). Random splits would place "
                f"related rows on both sides of the fold and produce optimistic validation. "
                f"{reason_extra}"
            ),
            warnings=[
                f"Pass the group column ({profile.likely_group_columns[0]}) via the "
                "`groups=` argument to `.split()` — the splitter cannot detect it on its own."
            ],
        )

    # 3) Classification with imbalance ----------------------------------
    if profile.task_type in ("binary", "multiclass"):
        if profile.imbalance_severity in ("moderate", "severe"):
            return ValidationRecommendation(
                strategy="StratifiedKFold",
                n_splits=5,
                reason=(
                    f"Classification target with {profile.imbalance_severity} imbalance. "
                    "StratifiedKFold preserves class ratios in each fold so the "
                    "validation metric is stable across folds."
                ),
                warnings=[
                    "Do not use accuracy as the CV metric on this imbalance level — "
                    "log loss, AUC, or PR AUC will discriminate better between models."
                ],
            )
        else:
            return ValidationRecommendation(
                strategy="StratifiedKFold",
                n_splits=5,
                reason=(
                    f"Classification target with {profile.imbalance_severity} imbalance. "
                    "StratifiedKFold is the safe default; keeps class ratios balanced "
                    "across folds even for mostly-balanced data."
                ),
            )

    # 4) Regression / unknown --------------------------------------------
    return ValidationRecommendation(
        strategy="KFold",
        n_splits=5,
        reason=(
            "Regression or unspecified task with no temporal or group structure detected. "
            "Standard KFold with shuffling gives a stable estimate."
        ),
        warnings=[
            "Set shuffle=True and a fixed random_state so folds are reproducible."
        ],
    )


def _from_dialogue_only(dialogue_context: dict[str, str]) -> ValidationRecommendation:
    """Fallback when no dataset was uploaded."""
    task = dialogue_context.get("task_type", "")
    if "regression" in task.lower():
        return ValidationRecommendation(
            strategy="KFold",
            n_splits=5,
            reason=(
                "No dataset uploaded. Regression task by dialogue — KFold with shuffle "
                "is the safe default. Switch to TimeSeriesSplit if the target has "
                "temporal order, or GroupKFold if rows are grouped."
            ),
        )
    if "classification" in task.lower():
        return ValidationRecommendation(
            strategy="StratifiedKFold",
            n_splits=5,
            reason=(
                "No dataset uploaded. Classification task by dialogue — StratifiedKFold "
                "is the safe default. Upload the CSV for a more specific recommendation."
            ),
        )
    return ValidationRecommendation(
        strategy="StratifiedKFold",
        n_splits=5,
        reason=(
            "No dataset uploaded and task type unclear. StratifiedKFold is a defensible "
            "default; revisit after uploading the CSV or clarifying the task."
        ),
    )
