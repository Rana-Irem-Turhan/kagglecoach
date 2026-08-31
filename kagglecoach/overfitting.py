"""
Rule-based overfitting risk assessment.

Two signals:
    * Dataset characteristics — small dataset + high feature count is
      inherently risky, so a warning fires proactively without needing
      any train/val scores.
    * Performance gap — when the user provides train and validation
      scores, the gap is classified into healthy / mild / moderate /
      severe using thresholds from `config.toml`.

The output is a `RiskAssessment` with a level and a list of specific
signals — no LLM involved. The LLM step later reads this structure and
renders it into prose alongside retrieved evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from kagglecoach.settings import SETTINGS, Settings


RISK_LEVELS = ("low", "medium", "high")


@dataclass
class RiskAssessment:
    level: str                              # 'low' | 'medium' | 'high'
    signals: list[str] = field(default_factory=list)
    gap_classification: Optional[str] = None  # 'healthy'|'mild'|'moderate'|'severe'|None
    numeric_gap: Optional[float] = None
    rows_per_feature: Optional[float] = None

    @property
    def is_actionable(self) -> bool:
        return self.level != "low"


def assess_overfitting(
    *,
    n_rows: int,
    n_features: int,
    train_score: Optional[float] = None,
    val_score: Optional[float] = None,
    higher_is_better: bool = True,
    settings: Settings = SETTINGS,
) -> RiskAssessment:
    """
    Assess overfitting risk from dataset shape and (optionally) train/val scores.

    If `train_score` and `val_score` are provided, gap severity is computed
    and combined with dataset characteristics into an overall level. Otherwise
    the assessment is based on dataset characteristics alone.
    """
    signals: list[str] = []
    gap_class: Optional[str] = None
    numeric_gap: Optional[float] = None

    # --- 1) Dataset-shape signals -------------------------------------
    rows_per_feature = n_rows / max(n_features, 1)
    if n_features == 0:
        rows_per_feature = float("inf")

    if rows_per_feature < settings.rows_per_feature_risky:
        signals.append(
            f"Only ~{rows_per_feature:.0f} rows per feature "
            f"(< {settings.rows_per_feature_risky} threshold). "
            "Complex models are prone to overfitting at this ratio."
        )

    if n_rows < 1000:
        signals.append(
            f"Very small dataset ({n_rows} rows). "
            "Prefer simple model families (linear, small tree ensembles)."
        )
    elif n_rows < 10_000:
        signals.append(
            f"Small dataset ({n_rows} rows). "
            "Regularise aggressively and cross-validate on 5+ folds."
        )

    # --- 2) Train/val gap signal --------------------------------------
    if train_score is not None and val_score is not None:
        if higher_is_better:
            numeric_gap = float(train_score) - float(val_score)
        else:
            numeric_gap = float(val_score) - float(train_score)

        # A negative gap means validation is BETTER than train — unusual,
        # sometimes indicates a data pipeline bug, but not overfitting.
        if numeric_gap < 0:
            signals.append(
                f"Validation score is better than training (gap {numeric_gap:+.4f}). "
                "This is unusual; audit the pipeline for label mixups or data leaks."
            )
            gap_class = "healthy"
        elif numeric_gap <= settings.gap_healthy_max:
            gap_class = "healthy"
            signals.append(
                f"Train/val gap {numeric_gap:.4f} is within the healthy range "
                f"(<= {settings.gap_healthy_max})."
            )
        elif numeric_gap <= settings.gap_mild_max:
            gap_class = "mild"
            signals.append(
                f"Train/val gap {numeric_gap:.4f} indicates mild overfitting "
                f"({settings.gap_healthy_max} < gap <= {settings.gap_mild_max})."
            )
        else:
            gap_class = "severe"
            signals.append(
                f"Train/val gap {numeric_gap:.4f} indicates substantial overfitting "
                f"(> {settings.gap_mild_max}). Audit before further tuning."
            )

    # --- 3) Combine into a single level -------------------------------
    level = _combine_level(gap_class, rows_per_feature, settings)

    if not signals:
        signals.append("No overfitting risk signals detected from dataset shape alone. "
                       "Provide train/validation scores for a stronger diagnosis.")

    return RiskAssessment(
        level=level,
        signals=signals,
        gap_classification=gap_class,
        numeric_gap=numeric_gap,
        rows_per_feature=rows_per_feature if rows_per_feature != float("inf") else None,
    )


def _combine_level(
    gap_class: Optional[str],
    rows_per_feature: float,
    settings: Settings,
) -> str:
    """Deterministic combination table."""
    # Severe gap always wins.
    if gap_class == "severe":
        return "high"

    # Small rows-per-feature ratio raises the base risk regardless of gap.
    dataset_risky = rows_per_feature < settings.rows_per_feature_risky

    if gap_class == "mild":
        return "high" if dataset_risky else "medium"

    if gap_class == "healthy":
        return "medium" if dataset_risky else "low"

    # No gap info supplied — go on dataset shape only.
    return "medium" if dataset_risky else "low"
