# Validation strategies for competitive ML

## Why validation strategy is the highest-impact decision

The validation strategy determines what your metric on held-out data actually measures. A validation setup that matches the competition's evaluation protocol gives you a reliable signal to iterate against; one that doesn't produces CV scores that drift from the leaderboard and misleads every subsequent decision.

Get validation right first. Every hour spent on model tuning against a bad CV is wasted.

## Decision tree: which strategy fits

Answer these in order:

**1. Does the target have a strong temporal component?**

Yes → TimeSeriesSplit (see below). No random splits.

No → continue.

**2. Are rows grouped (multiple rows per user, patient, session, article)?**

Yes → GroupKFold or StratifiedGroupKFold. Groups must not cross fold boundaries.

No → continue.

**3. Is the target classification with imbalance?**

Yes → StratifiedKFold. Each fold preserves class ratios.

No → KFold with `shuffle=True` and a fixed seed.

**4. Is the test set from a meaningfully different distribution than training?**

Yes → consider adversarial validation to weight or subsample training rows toward test distribution.

## StratifiedKFold — the workhorse

Standard for classification without temporal or group structure:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    # train, evaluate, store fold predictions
```

5 folds is standard. 10 folds gives lower-variance estimates at 2x compute; use it for the final tuning push, not for iteration.

## TimeSeriesSplit — non-negotiable for temporal problems

Any competition where the target has time ordering must use temporal validation. Random splits let the model see the future during training:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
```

Two important properties:

- Training data is always earlier than validation data within a fold.
- Training data grows across folds (expanding window). If you need equal training-set size across folds, set `max_train_size=`.

**Common temporal validation mistakes:**

- Sorting by an ID column and using KFold. If IDs are time-correlated, the fold structure is arbitrarily temporal — the model may or may not leak the future.
- Feature engineering with future information. Aggregations, rolling means, and target encoding must use only past rows within each fold.
- Ignoring the gap between train and test. If the competition test set is from 2 months after training, use `TimeSeriesSplit(gap=)` to simulate that gap in your validation.

## GroupKFold — for grouped data

When multiple rows share a group (user_id, patient_id, article_id), a random split places related rows in both train and test. The model learns group-specific patterns and looks great on validation but fails on held-out groups.

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for tr_idx, val_idx in gkf.split(X, y, groups=user_ids):
    # each user appears in exactly one fold
    ...
```

`StratifiedGroupKFold` (scikit-learn 1.0+) adds stratification when the group structure is compatible with the target distribution.

**Detecting group structure:** if any column has many rows per unique value and could plausibly identify a common subject, that column is a group candidate. Check by computing `df[col].value_counts()` — long tails of repeated values indicate potential grouping.

## Nested CV — for unbiased estimates

If you tune hyperparameters against a single CV and report the best score, the reported number is optimistic. You selected the winner of a lottery.

Nested CV has two loops: outer for evaluation, inner for tuning:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(model, param_grid, cv=inner_cv)
scores = cross_val_score(search, X, y, cv=outer_cv)
print(f"Unbiased estimate: {scores.mean():.4f}")
```

Cost is `outer_k * inner_k * len(param_grid)` fits — expensive. Use for reporting final numbers, not for iteration.

## Adversarial validation

When train and test distributions differ meaningfully (drift, sampling difference, temporal shift), CV score misleads. Adversarial validation quantifies the difference and can guide sample weighting.

Steps:

1. Combine train and test into one dataset; label rows with `is_test = 0/1`.
2. Train a classifier to distinguish train from test using the features (not the target).
3. If AUC is near 0.5, distributions match — standard CV is fine.
4. If AUC is high (>0.8), distributions differ substantially. Use the classifier's predicted probability of being test as a sample weight during training, or use it to select training rows most similar to test.

Adversarial validation also flags leaky features: any single feature that hugely helps distinguish train from test is either an identifier (drop it) or a temporally-shifted feature (investigate).

## The CV vs. leaderboard gap

A stable local-to-public gap that persists across submissions is fine. Trust local CV for iteration; use the leaderboard as a sanity check.

Gap grows or shrinks between submissions? Almost always means a leakage issue in newly-added features. Audit them.

Gap is enormous (>0.05 on AUC scale) from the start? Your validation setup is wrong for this competition. Revisit the decision tree above.

## Common failure modes

- **Random split on time-series data.** Model sees the future. Symptoms: perfect CV, terrible leaderboard.
- **Preprocessing outside CV.** Fitting a scaler or encoder on the full dataset before splitting. Symptoms: mild CV inflation that vanishes on the leaderboard.
- **Insufficient folds.** 3 folds gives noisy estimates. Use 5 or 10.
- **Fold count vs. dataset size mismatch.** 10 folds on 1000 rows leaves 100 rows per validation — noise dominates. Reduce folds or accept the noise floor.
- **Meta-model with non-OOF Level 0.** Stacking without out-of-fold base predictions produces overfit meta-models. Symptoms: enormous CV/leaderboard gap on stacked submissions.
