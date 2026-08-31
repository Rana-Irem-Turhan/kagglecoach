"""
Pandas-based dataset profiler.

Given a DataFrame (or a CSV path), produces a `DatasetProfile` — a
structured summary of the dataset's shape, dtypes, missingness,
cardinality, target distribution, and likely group columns.

This layer is pure pandas/numpy. No LLM involved. The profile feeds
directly into strategy selection later — grounded, deterministic,
testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from kagglecoach.settings import SETTINGS, Settings


# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------
@dataclass
class DatasetProfile:
    """Structured summary of a dataset."""

    # Shape
    n_rows: int
    n_columns: int
    size_bucket: str                        # 'small' | 'medium' | 'large' | 'very_large'
    memory_usage_mb: float

    # Target
    target_column: Optional[str]
    task_type: str                          # 'binary' | 'multiclass' | 'regression' | 'unknown'
    n_classes: Optional[int]
    class_balance: Optional[dict[str, float]]
    imbalance_severity: str                 # 'balanced' | 'mild' | 'moderate' | 'severe' | 'n/a'

    # Column types
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    id_like_columns: list[str] = field(default_factory=list)

    # Missingness
    missing_rates: dict[str, float] = field(default_factory=dict)
    columns_with_moderate_missing: list[str] = field(default_factory=list)
    columns_with_severe_missing: list[str] = field(default_factory=list)

    # Cardinality
    cardinality: dict[str, int] = field(default_factory=dict)
    high_cardinality_categorical: list[str] = field(default_factory=list)

    # Grouping candidates
    likely_group_columns: list[str] = field(default_factory=list)

    # Correlations (numeric only, if target is numeric or binary)
    numeric_correlations_with_target: dict[str, float] = field(default_factory=dict)

    # Textual summary (for the report)
    summary_bullets: list[str] = field(default_factory=list)

    @property
    def rows_per_feature(self) -> float:
        return self.n_rows / max(self.n_columns - 1, 1)


# --------------------------------------------------------------------------
# Profiler
# --------------------------------------------------------------------------
def profile_dataframe(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    settings: Settings = SETTINGS,
) -> DatasetProfile:
    """Compute a full profile of `df`."""
    if df is None or df.empty:
        raise ValueError("Cannot profile an empty DataFrame.")

    n_rows, n_columns = df.shape
    size_bucket = _size_bucket(n_rows, settings)

    # Column categorisation --------------------------------------------
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    text_cols: list[str] = []
    id_like: list[str] = []

    for col in df.columns:
        series = df[col]
        if col == target_column:
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_cols.append(col)
            if _looks_like_id_column(series, col):
                id_like.append(col)
            continue

        # Object / string / category — decide between short-categorical, text, or id-like.
        n_unique = series.nunique(dropna=True)
        n_non_null = series.notna().sum()
        avg_length = _average_string_length(series)

        if avg_length is not None and avg_length > 40 and n_unique / max(n_non_null, 1) > 0.5:
            # Long, mostly-unique strings — treat as text.
            text_cols.append(col)
        elif _looks_like_id_column(series, col):
            id_like.append(col)
        else:
            categorical_cols.append(col)

    # Missingness ------------------------------------------------------
    missing_rates = {col: float(df[col].isna().mean()) for col in df.columns}
    moderate = [
        c for c, r in missing_rates.items()
        if settings.missing_moderate <= r < settings.missing_severe
    ]
    severe = [c for c, r in missing_rates.items() if r >= settings.missing_severe]

    # Cardinality ------------------------------------------------------
    cardinality = {
        col: int(df[col].nunique(dropna=True))
        for col in categorical_cols + text_cols + id_like
    }
    high_card = [
        c for c in categorical_cols
        if cardinality.get(c, 0) >= settings.high_cardinality_threshold
    ]

    # Target analysis --------------------------------------------------
    task_type = "unknown"
    n_classes: Optional[int] = None
    class_balance: Optional[dict[str, float]] = None
    imbalance_severity = "n/a"
    numeric_corr: dict[str, float] = {}

    if target_column is not None and target_column in df.columns:
        y = df[target_column].dropna()
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
            task_type = "regression"
        else:
            unique = y.nunique()
            if unique == 2:
                task_type = "binary"
                n_classes = 2
            elif 2 < unique <= 50:
                task_type = "multiclass"
                n_classes = int(unique)
            else:
                task_type = "unknown"

            if task_type in ("binary", "multiclass"):
                counts = y.value_counts(normalize=True)
                class_balance = {str(k): float(v) for k, v in counts.items()}
                majority = max(class_balance.values())
                imbalance_severity = _imbalance_severity(majority)

        # Numeric correlations with target (works for regression and binary).
        if task_type in ("regression", "binary") and pd.api.types.is_numeric_dtype(df[target_column]):
            for col in numeric_cols:
                try:
                    c = df[[col, target_column]].dropna()
                    if len(c) < 30:
                        continue
                    corr = float(c[col].corr(c[target_column]))
                    if not np.isnan(corr):
                        numeric_corr[col] = corr
                except Exception:  # noqa: BLE001
                    continue

    # Group column candidates ------------------------------------------
    group_candidates = _detect_group_columns(df, categorical_cols + id_like)

    # Build summary bullets --------------------------------------------
    bullets: list[str] = []
    bullets.append(
        f"{n_rows:,} rows × {n_columns} columns "
        f"({size_bucket} dataset by our size buckets)"
    )
    if target_column:
        if task_type == "regression":
            bullets.append(f"Target `{target_column}`: regression")
        elif task_type in ("binary", "multiclass") and class_balance:
            majority = max(class_balance.values())
            bullets.append(
                f"Target `{target_column}`: {task_type} classification with "
                f"{n_classes} class(es); majority {majority:.1%} — {imbalance_severity} imbalance"
            )
        else:
            bullets.append(f"Target `{target_column}`: task type unclear")

    if datetime_cols:
        bullets.append(f"Datetime columns detected: {', '.join(datetime_cols)} "
                       f"(consider temporal validation)")
    if high_card:
        bullets.append(
            f"High-cardinality categoricals (>= {settings.high_cardinality_threshold} unique): "
            f"{', '.join(high_card)} (target encoding candidates)"
        )
    if severe:
        bullets.append(f"Severe missingness (>= {settings.missing_severe:.0%}): {', '.join(severe)}")
    elif moderate:
        bullets.append(f"Moderate missingness (>= {settings.missing_moderate:.0%}): "
                       f"{', '.join(moderate[:5])}"
                       + (f" and {len(moderate) - 5} more" if len(moderate) > 5 else ""))
    if group_candidates:
        bullets.append(
            f"Likely group columns (potential group leakage risk): "
            f"{', '.join(group_candidates)}"
        )
    if text_cols:
        bullets.append(f"Text columns detected: {', '.join(text_cols)}")

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=n_columns,
        size_bucket=size_bucket,
        memory_usage_mb=float(df.memory_usage(deep=True).sum()) / (1024 * 1024),
        target_column=target_column,
        task_type=task_type,
        n_classes=n_classes,
        class_balance=class_balance,
        imbalance_severity=imbalance_severity,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        text_columns=text_cols,
        id_like_columns=id_like,
        missing_rates=missing_rates,
        columns_with_moderate_missing=moderate,
        columns_with_severe_missing=severe,
        cardinality=cardinality,
        high_cardinality_categorical=high_card,
        likely_group_columns=group_candidates,
        numeric_correlations_with_target=numeric_corr,
        summary_bullets=bullets,
    )


def profile_csv(
    path: Path | str,
    target_column: Optional[str] = None,
    settings: Settings = SETTINGS,
    sample_rows: Optional[int] = None,
) -> DatasetProfile:
    """Load a CSV (optionally sampled) and profile it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    read_kwargs: dict = {"low_memory": False}
    if sample_rows is not None:
        read_kwargs["nrows"] = sample_rows

    df = pd.read_csv(path, **read_kwargs)

    # Attempt datetime parsing on object columns whose values look like dates.
    for col in df.columns:
        if df[col].dtype == object and _looks_like_datetime(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:  # noqa: BLE001
                pass

    return profile_dataframe(df, target_column=target_column, settings=settings)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _size_bucket(n_rows: int, settings: Settings) -> str:
    if n_rows < settings.size_small_max:
        return "small"
    if n_rows < settings.size_medium_max:
        return "medium"
    if n_rows < settings.size_large_max:
        return "large"
    return "very_large"


def _imbalance_severity(majority_fraction: float) -> str:
    if majority_fraction < 0.60:
        return "balanced"
    if majority_fraction < 0.80:
        return "mild"
    if majority_fraction < 0.95:
        return "moderate"
    return "severe"


def _looks_like_id_column(series: pd.Series, name: str) -> bool:
    """Heuristic for 'this column is an identifier, not a feature'."""
    lowered = name.lower()
    if lowered.endswith("_id") or lowered == "id":
        # Almost certainly an ID.
        return True
    n_unique = series.nunique(dropna=True)
    n_non_null = series.notna().sum()
    # Column with near-unique values across rows — likely an ID.
    if n_non_null > 0 and n_unique / n_non_null > 0.98 and n_unique > 100:
        return True
    return False


def _average_string_length(series: pd.Series) -> Optional[float]:
    """Average length of stringified non-null values, or None if impossible."""
    non_null = series.dropna()
    if non_null.empty:
        return None
    try:
        lengths = non_null.astype(str).str.len()
        return float(lengths.mean())
    except Exception:  # noqa: BLE001
        return None


def _looks_like_datetime(series: pd.Series, sample_size: int = 50) -> bool:
    """Sample a few string values and see if they parse as dates."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    sample = non_null.head(sample_size)
    parsed = pd.to_datetime(sample, errors="coerce")
    if parsed.notna().mean() < 0.9:
        return False
    # Guard against pure integers (which pandas happily parses as epoch seconds).
    if sample.str.match(r"^\d+$").mean() > 0.5:
        return False
    return True


def _detect_group_columns(
    df: pd.DataFrame,
    candidates: list[str],
) -> list[str]:
    """
    Columns that repeat across many rows (>3 rows per unique value on average
    and unique count between 10 and n_rows/3) are group candidates.
    """
    out: list[str] = []
    n_rows = len(df)
    for col in candidates:
        n_unique = df[col].nunique(dropna=True)
        if n_unique < 10 or n_unique > n_rows / 3:
            continue
        avg_rows_per_group = n_rows / n_unique
        if avg_rows_per_group >= 3:
            out.append(col)
    return out
