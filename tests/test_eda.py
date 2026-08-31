"""Tests for `kagglecoach.eda` — pandas dataset profiler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kagglecoach.eda import DatasetProfile, profile_dataframe


# --------------------------------------------------------------------------
# Shape and size buckets
# --------------------------------------------------------------------------
def test_size_bucket_small():
    df = pd.DataFrame({"x": range(500), "y": range(500)})
    p = profile_dataframe(df, target_column="y")
    assert p.size_bucket == "small"


def test_size_bucket_medium():
    df = pd.DataFrame({"x": range(50_000), "y": range(50_000)})
    p = profile_dataframe(df, target_column="y")
    assert p.size_bucket == "medium"


def test_size_bucket_large():
    df = pd.DataFrame({"x": range(500_000), "y": range(500_000)})
    p = profile_dataframe(df, target_column="y")
    assert p.size_bucket == "large"


# --------------------------------------------------------------------------
# Column typing
# --------------------------------------------------------------------------
def test_numeric_and_categorical_columns_detected():
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "city": ["NYC", "LA", "NYC", "SF", "LA"],
        "target": [0, 1, 0, 1, 0],
    })
    p = profile_dataframe(df, target_column="target")
    assert "age" in p.numeric_columns
    assert "city" in p.categorical_columns
    assert "target" not in p.numeric_columns  # excluded because it's the target


def test_id_column_detected_by_name():
    df = pd.DataFrame({
        "user_id": range(200),
        "amount": np.random.rand(200),
        "target": np.random.randint(0, 2, 200),
    })
    p = profile_dataframe(df, target_column="target")
    assert "user_id" in p.id_like_columns


def test_id_column_detected_by_uniqueness():
    df = pd.DataFrame({
        # Nearly unique object column but no _id suffix
        "hash": [f"h{i:05d}" for i in range(500)],
        "amount": np.random.rand(500),
        "target": np.random.randint(0, 2, 500),
    })
    p = profile_dataframe(df, target_column="target")
    assert "hash" in p.id_like_columns


def test_text_column_detected():
    df = pd.DataFrame({
        "review": ["This is a really long piece of text describing the product in detail." * 2] * 50,
        "target": [0, 1] * 25,
    })
    # Vary the text a bit so uniqueness is high
    df["review"] = df["review"] + [f" variant {i}" for i in range(50)]
    p = profile_dataframe(df, target_column="target")
    assert "review" in p.text_columns


def test_datetime_column_detected():
    df = pd.DataFrame({
        "ts": pd.date_range("2023-01-01", periods=100, freq="D"),
        "value": np.random.rand(100),
        "target": np.random.randint(0, 2, 100),
    })
    p = profile_dataframe(df, target_column="target")
    assert "ts" in p.datetime_columns


# --------------------------------------------------------------------------
# Target analysis
# --------------------------------------------------------------------------
def test_binary_classification_detected():
    df = pd.DataFrame({
        "x": np.random.rand(200),
        "target": np.random.randint(0, 2, 200),
    })
    p = profile_dataframe(df, target_column="target")
    assert p.task_type == "binary"
    assert p.n_classes == 2


def test_multiclass_classification_detected():
    df = pd.DataFrame({
        "x": np.random.rand(300),
        "target": np.random.randint(0, 5, 300),
    })
    p = profile_dataframe(df, target_column="target")
    assert p.task_type == "multiclass"
    assert p.n_classes == 5


def test_regression_detected():
    df = pd.DataFrame({
        "x": np.random.rand(300),
        "target": np.random.rand(300) * 100,
    })
    p = profile_dataframe(df, target_column="target")
    assert p.task_type == "regression"


def test_imbalance_severity_severe():
    df = pd.DataFrame({
        "x": np.random.rand(1000),
        "target": [0] * 990 + [1] * 10,
    })
    p = profile_dataframe(df, target_column="target")
    assert p.imbalance_severity == "severe"


def test_imbalance_severity_balanced():
    df = pd.DataFrame({
        "x": np.random.rand(200),
        "target": [0] * 100 + [1] * 100,
    })
    p = profile_dataframe(df, target_column="target")
    assert p.imbalance_severity == "balanced"


# --------------------------------------------------------------------------
# Missingness and cardinality
# --------------------------------------------------------------------------
def test_severe_missing_flagged():
    df = pd.DataFrame({
        "mostly_missing": [np.nan] * 90 + list(range(10)),  # 90% missing
        "target": np.random.randint(0, 2, 100),
    })
    p = profile_dataframe(df, target_column="target")
    assert "mostly_missing" in p.columns_with_severe_missing


def test_high_cardinality_flagged():
    df = pd.DataFrame({
        "many_categories": [f"cat_{i % 500}" for i in range(2000)],
        "target": np.random.randint(0, 2, 2000),
    })
    p = profile_dataframe(df, target_column="target")
    # 500 categories >= 100 threshold
    assert "many_categories" in p.high_cardinality_categorical


# --------------------------------------------------------------------------
# Group detection
# --------------------------------------------------------------------------
def test_likely_group_column_detected():
    df = pd.DataFrame({
        # 20 unique users × 10 rows each = 200 rows
        "user": [f"u{i}" for i in range(20) for _ in range(10)],
        "value": np.random.rand(200),
        "target": np.random.randint(0, 2, 200),
    })
    p = profile_dataframe(df, target_column="target")
    assert "user" in p.likely_group_columns


def test_no_group_when_all_unique():
    df = pd.DataFrame({
        "unique_col": [f"u{i}" for i in range(200)],
        "target": np.random.randint(0, 2, 200),
    })
    p = profile_dataframe(df, target_column="target")
    # unique_col is id-like or high-cardinality — should not be group
    assert "unique_col" not in p.likely_group_columns


# --------------------------------------------------------------------------
# Summary bullets
# --------------------------------------------------------------------------
def test_summary_bullets_populated():
    df = pd.DataFrame({
        "x": np.random.rand(1000),
        "target": np.random.randint(0, 2, 1000),
    })
    p = profile_dataframe(df, target_column="target")
    assert p.summary_bullets  # not empty
    # First bullet should always mention shape
    assert "rows" in p.summary_bullets[0]


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------
def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        profile_dataframe(pd.DataFrame())


def test_no_target_still_profiles():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    p = profile_dataframe(df)  # no target
    assert p.task_type == "unknown"
    assert p.n_rows == 100


def test_rows_per_feature_computed():
    df = pd.DataFrame({f"c{i}": np.random.rand(500) for i in range(50)})
    df["target"] = np.random.randint(0, 2, 500)
    p = profile_dataframe(df, target_column="target")
    # 500 rows, 50 features → 10 rows per feature
    assert 9 <= p.rows_per_feature <= 11
