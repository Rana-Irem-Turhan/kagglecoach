"""Tests for `kagglecoach.overfitting`."""

from __future__ import annotations

from kagglecoach.overfitting import assess_overfitting


# --------------------------------------------------------------------------
# No performance scores → dataset-shape reasoning only
# --------------------------------------------------------------------------
def test_low_risk_large_dataset_no_scores():
    r = assess_overfitting(n_rows=1_000_000, n_features=20)
    assert r.level == "low"
    assert r.gap_classification is None


def test_medium_risk_very_few_rows_per_feature():
    r = assess_overfitting(n_rows=500, n_features=50)  # 10 rows/feature
    assert r.level == "medium"
    assert any("rows per feature" in s for s in r.signals)


def test_medium_risk_very_small_dataset():
    r = assess_overfitting(n_rows=800, n_features=5)  # 160 rows/feat but tiny
    # tiny dataset warning fires
    assert any("small" in s.lower() for s in r.signals)


# --------------------------------------------------------------------------
# With train/val scores
# --------------------------------------------------------------------------
def test_healthy_gap_and_large_dataset_is_low_risk():
    r = assess_overfitting(
        n_rows=200_000, n_features=30,
        train_score=0.85, val_score=0.83,
    )
    assert r.level == "low"
    assert r.gap_classification == "healthy"


def test_mild_gap_medium_dataset_is_medium():
    r = assess_overfitting(
        n_rows=200_000, n_features=30,
        train_score=0.90, val_score=0.85,  # gap 0.05
    )
    assert r.level == "medium"
    assert r.gap_classification == "mild"


def test_severe_gap_always_high():
    r = assess_overfitting(
        n_rows=200_000, n_features=30,
        train_score=0.98, val_score=0.75,  # gap 0.23
    )
    assert r.level == "high"
    assert r.gap_classification == "severe"


def test_mild_gap_on_risky_dataset_becomes_high():
    r = assess_overfitting(
        n_rows=500, n_features=50,   # 10 rows/feat → risky
        train_score=0.90, val_score=0.85,
    )
    assert r.level == "high"


def test_healthy_gap_on_risky_dataset_becomes_medium():
    r = assess_overfitting(
        n_rows=500, n_features=50,
        train_score=0.85, val_score=0.83,
    )
    assert r.level == "medium"


# --------------------------------------------------------------------------
# Direction of the metric
# --------------------------------------------------------------------------
def test_lower_is_better_flips_gap_sign():
    """For loss metrics, val > train means overfitting."""
    r = assess_overfitting(
        n_rows=200_000, n_features=30,
        train_score=0.10, val_score=0.25,      # val loss > train loss
        higher_is_better=False,
    )
    # gap = 0.25 - 0.10 = 0.15 → severe
    assert r.gap_classification == "severe"


def test_negative_gap_reports_data_bug():
    """Val > train (for higher_is_better) is unusual, not overfitting."""
    r = assess_overfitting(
        n_rows=200_000, n_features=30,
        train_score=0.70, val_score=0.85,
    )
    assert r.gap_classification == "healthy"
    assert any("unusual" in s.lower() or "leak" in s.lower() for s in r.signals)


# --------------------------------------------------------------------------
# Signal ordering
# --------------------------------------------------------------------------
def test_signals_always_populated():
    r = assess_overfitting(n_rows=1_000_000, n_features=10)
    assert r.signals  # not empty


def test_numeric_gap_recorded_when_scores_supplied():
    r = assess_overfitting(
        n_rows=200_000, n_features=30,
        train_score=0.90, val_score=0.85,
    )
    assert r.numeric_gap == pytest_approx_equal(0.05)


def pytest_approx_equal(value: float, tol: float = 1e-6):
    """Tiny approximation helper without importing pytest.approx."""
    class _A:
        def __eq__(self, other): return abs(other - value) < tol
    return _A()
