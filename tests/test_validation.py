"""Tests for `kagglecoach.validation`."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kagglecoach.eda import profile_dataframe
from kagglecoach.validation import recommend_validation


# --------------------------------------------------------------------------
# Datetime → TimeSeriesSplit
# --------------------------------------------------------------------------
def test_datetime_column_triggers_timeseriessplit():
    df = pd.DataFrame({
        "ts": pd.date_range("2023-01-01", periods=1000, freq="D"),
        "x": np.random.rand(1000),
        "target": np.random.randint(0, 2, 1000),
    })
    profile = profile_dataframe(df, target_column="target")
    rec = recommend_validation(profile, {})
    assert rec.strategy == "TimeSeriesSplit"
    assert "temporal" in rec.reason.lower() or "time" in rec.reason.lower()


def test_timeseriessplit_warns_about_rolling_features():
    df = pd.DataFrame({
        "ts": pd.date_range("2023-01-01", periods=500, freq="D"),
        "x": np.random.rand(500),
        "target": np.random.randint(0, 2, 500),
    })
    profile = profile_dataframe(df, target_column="target")
    rec = recommend_validation(profile, {})
    assert any("rolling" in w.lower() or "target encoding" in w.lower()
               for w in rec.warnings)


# --------------------------------------------------------------------------
# Group column → GroupKFold / StratifiedGroupKFold
# --------------------------------------------------------------------------
def test_group_column_triggers_stratified_group_for_classification():
    df = pd.DataFrame({
        "user": [f"u{i}" for i in range(20) for _ in range(10)],  # 20 groups × 10 rows
        "value": np.random.rand(200),
        "target": np.random.randint(0, 2, 200),
    })
    profile = profile_dataframe(df, target_column="target")
    rec = recommend_validation(profile, {})
    assert rec.strategy == "StratifiedGroupKFold"


def test_group_column_triggers_plain_groupkfold_for_regression():
    df = pd.DataFrame({
        "user": [f"u{i}" for i in range(20) for _ in range(10)],
        "value": np.random.rand(200),
        "target": np.random.rand(200) * 100,   # regression target
    })
    profile = profile_dataframe(df, target_column="target")
    rec = recommend_validation(profile, {})
    assert rec.strategy == "GroupKFold"


def test_group_warning_mentions_groups_argument():
    df = pd.DataFrame({
        "user": [f"u{i}" for i in range(20) for _ in range(10)],
        "value": np.random.rand(200),
        "target": np.random.randint(0, 2, 200),
    })
    profile = profile_dataframe(df, target_column="target")
    rec = recommend_validation(profile, {})
    assert any("groups=" in w or "group column" in w.lower() for w in rec.warnings)


# --------------------------------------------------------------------------
# Classification imbalance → StratifiedKFold
# --------------------------------------------------------------------------
def test_severe_imbalance_triggers_stratifiedkfold():
    df = pd.DataFrame({
        "x": np.random.rand(1000),
        "target": [0] * 970 + [1] * 30,
    })
    profile = profile_dataframe(df, target_column="target")
    rec = recommend_validation(profile, {})
    assert rec.strategy == "StratifiedKFold"
    # Warns off accuracy
    assert any("accuracy" in w.lower() for w in rec.warnings)


def test_balanced_classification_still_stratified():
    df = pd.DataFrame({
        "x": np.random.rand(500),
        "target": [0] * 250 + [1] * 250,
    })
    profile = profile_dataframe(df, target_column="target")
    rec = recommend_validation(profile, {})
    assert rec.strategy == "StratifiedKFold"


# --------------------------------------------------------------------------
# Regression → KFold
# --------------------------------------------------------------------------
def test_regression_triggers_kfold():
    df = pd.DataFrame({
        "x": np.random.rand(500),
        "target": np.random.rand(500) * 100,
    })
    profile = profile_dataframe(df, target_column="target")
    rec = recommend_validation(profile, {})
    assert rec.strategy == "KFold"


# --------------------------------------------------------------------------
# No profile → dialogue-only fallback
# --------------------------------------------------------------------------
def test_no_profile_regression_fallback():
    rec = recommend_validation(None, {"task_type": "Tabular regression"})
    assert rec.strategy == "KFold"


def test_no_profile_classification_fallback():
    rec = recommend_validation(None, {"task_type": "Tabular classification"})
    assert rec.strategy == "StratifiedKFold"


def test_no_profile_unknown_fallback():
    rec = recommend_validation(None, {})
    assert rec.strategy == "StratifiedKFold"


# --------------------------------------------------------------------------
# n_splits is always sensible
# --------------------------------------------------------------------------
def test_n_splits_default():
    rec = recommend_validation(None, {"task_type": "Tabular classification"})
    assert 3 <= rec.n_splits <= 10
