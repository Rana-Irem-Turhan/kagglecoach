# Feature engineering patterns for tabular competitions

## The three-tier framework

Feature engineering in competition tabular problems separates into three tiers based on effort-vs-reward:

**Tier 1 — Cheap and reliable.** Run these on every problem:

- Datetime decomposition (year, month, day, day_of_week, hour, is_weekend).
- Cardinality-based encoding selection (see below).
- Missing indicator columns for features with missing rates above 5%.
- Log or square-root transformation of right-skewed numeric features.
- Aggregations by natural group columns (mean, std, min, max, count).

**Tier 2 — Domain-informed.** Requires reading the competition description carefully:

- Cross-column interactions where domain suggests they matter (price × quantity → revenue, latitude × longitude → geospatial cluster).
- Ratios and differences (feature_A / feature_B for related features).
- Rolling statistics over time-sorted rows within a group.
- Encoding column combinations (target encoding of category_A + category_B pair).

**Tier 3 — Expensive and risky.** Only when Tier 1 and 2 are exhausted:

- Automated feature synthesis (featuretools, autofeat) — high risk of leakage, most generated features are noise.
- Neural network embeddings for high-cardinality categoricals extracted as features for the gradient boosting model.
- Feature interactions discovered via SHAP interaction values.
- Pseudo-labeling using model predictions on the test set as features (advanced, easy to leak).

Most competition-winning solutions use extensive Tier 1 and Tier 2 engineering. Tier 3 shows up in top-10 solutions but rarely accounts for more than half of the improvement over baseline.

## Encoding strategy by cardinality

Match encoding to cardinality, not to convenience:

| Cardinality | Best encoding | Notes |
|---|---|---|
| 2 (binary) | Label (0/1) | Trivial. |
| 3-10 | One-hot | Sparse but always works. |
| 10-50 | Native categorical (LGBM/CatBoost) or one-hot | One-hot is fine for tree models but bloats memory. |
| 50-1000 | Target encoding with out-of-fold folds | The single most impactful feature engineering choice for medium-cardinality problems. |
| 1000+ | CatBoost native, or frequency encoding, or hashing | High risk with target encoding; validate carefully. |

## Target encoding — the correct way

Naive target encoding leaks the target into the encoded feature. Fix it with out-of-fold computation:

```python
from sklearn.model_selection import KFold

def target_encode_oof(train, valid, col, target, n_splits=5, smoothing=10):
    """Compute target encoding on train using OOF folds; apply the full-train
    encoding to valid."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_encoded = pd.Series(index=train.index, dtype=float)
    global_mean = train[target].mean()

    for tr_idx, val_idx in kf.split(train):
        fold_train = train.iloc[tr_idx]
        fold_val = train.iloc[val_idx]
        # Smoothed mean per category
        stats = fold_train.groupby(col)[target].agg(["mean", "count"])
        stats["smoothed"] = (stats["mean"] * stats["count"] + global_mean * smoothing) \
                            / (stats["count"] + smoothing)
        train_encoded.iloc[val_idx] = fold_val[col].map(stats["smoothed"]).fillna(global_mean)

    # For valid, use encoding fit on the full train set
    stats_full = train.groupby(col)[target].agg(["mean", "count"])
    stats_full["smoothed"] = (stats_full["mean"] * stats_full["count"] + global_mean * smoothing) \
                             / (stats_full["count"] + smoothing)
    valid_encoded = valid[col].map(stats_full["smoothed"]).fillna(global_mean)
    return train_encoded, valid_encoded
```

Two rules that prevent leakage:

1. Never use rows from the same fold to encode themselves. The OOF loop above enforces this.
2. The encoding used at prediction time must be fit on the training data only, never on train + test.

## Datetime decomposition

Any datetime column carries multiple periodic signals. Extract them:

```python
df["year"] = df["ts"].dt.year
df["month"] = df["ts"].dt.month
df["day"] = df["ts"].dt.day
df["day_of_week"] = df["ts"].dt.dayofweek
df["hour"] = df["ts"].dt.hour
df["is_weekend"] = df["day_of_week"] >= 5
df["is_month_start"] = df["ts"].dt.is_month_start
df["is_holiday"] = df["ts"].isin(holiday_dates)   # domain-specific
```

For strongly periodic features (hour, month, day_of_week), sine/cosine encoding preserves the wrap-around structure:

```python
import numpy as np
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

Tree-based models often don't need the sine/cosine version (they can split around discrete hours). Linear and neural models benefit from it.

## Aggregation features

When rows have a natural group column (user_id, session_id, sku), aggregate numeric features by that group and join back:

```python
agg = df.groupby("user_id")["amount"].agg(["mean", "std", "min", "max", "sum", "count"])
agg.columns = [f"user_amount_{c}" for c in agg.columns]
df = df.merge(agg, left_on="user_id", right_index=True, how="left")
```

Two things to watch:

- If you're doing time-based prediction, ensure the aggregation uses only past rows (rolling / expanding windows). Aggregating over the full time range leaks the future into the past.
- Ratios of raw feature to group aggregate (`amount / user_amount_mean`) often carry more signal than either alone.

## Interaction features

Two categoricals combined into a composite key often reveal signal neither one exposes alone:

```python
df["state_x_category"] = df["state"].astype(str) + "_" + df["category"].astype(str)
```

Then target-encode the composite (with OOF folds). For tree models this can also be done by letting the model discover the interaction — but for high-cardinality pairs the explicit encoding often wins because it reduces the search space.

## What not to do

- **Do not use test-set statistics for imputation, scaling, or encoding.** This is target-adjacent leakage that inflates CV without helping the leaderboard.
- **Do not one-hot encode a >50-cardinality column.** It bloats memory and hurts tree models.
- **Do not add every possible interaction.** Adding noise features slows training and dilutes signal.
- **Do not skip validation-time features.** Every feature you compute on train must be reproducible on validation and test with the same code path.
