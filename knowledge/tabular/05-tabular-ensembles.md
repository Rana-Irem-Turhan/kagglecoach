# Ensembling strategies for tabular competitions

## Why ensembling wins

A single model captures one view of the data. An ensemble captures multiple views and averages out idiosyncratic errors. On tabular competitions, ensembling typically moves your score 0.001-0.010 on AUC-style metrics — often the difference between a bronze and a gold medal.

Two rules govern effective ensembling:

1. **Diversity matters more than individual model quality.** A weak but different model contributes more to the ensemble than a strong but correlated one.
2. **Ensembles help less as base models improve.** When your best model is already at the theoretical ceiling of a metric, averaging gains vanish.

## Blending: the simplest effective approach

Blending is a weighted average of model predictions. For classification, average the probabilities. For regression, average the predictions directly.

```python
# Simple average
blend_preds = (lgb_preds + xgb_preds + catboost_preds) / 3

# Weighted by validation score
weights = [0.4, 0.3, 0.3]   # tuned on validation
blend_preds = sum(w * p for w, p in zip(weights, [lgb_preds, xgb_preds, cat_preds]))
```

For most competitions, an equal average of 3-5 diverse strong models is 80% of the value at 20% of the complexity.

**Weight tuning.** Use scipy.optimize.minimize with the negative validation metric as the loss. Don't tune weights by hand — the search space is small enough that automated search finds better weights.

## Stacking: when blending isn't enough

Stacking uses out-of-fold predictions from base models as features for a meta-model. The meta-model learns to combine base predictions non-linearly.

```
Level 0: LightGBM, XGBoost, CatBoost, Random Forest, Linear Model
         → each produces OOF predictions on train
         → each produces predictions on test

Level 1: Meta-model (usually linear or logistic regression) trained on OOF predictions
         → predicts on test using base models' test predictions
```

**The OOF requirement is non-negotiable.** If you train base models on the full training set and use those predictions as meta-features, the meta-model overfits massively and generalizes poorly.

Sketch:

```python
from sklearn.model_selection import KFold

def make_oof(base_model, X, y, X_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    for tr_idx, val_idx in kf.split(X):
        base_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = base_model.predict_proba(X.iloc[val_idx])[:, 1]
        test_preds += base_model.predict_proba(X_test)[:, 1] / n_splits
    return oof, test_preds

# Level 0
lgb_oof, lgb_test = make_oof(lgb_model, X_train, y_train, X_test)
xgb_oof, xgb_test = make_oof(xgb_model, X_train, y_train, X_test)
cat_oof, cat_test = make_oof(cat_model, X_train, y_train, X_test)

# Level 1 meta-features
meta_train = np.column_stack([lgb_oof, xgb_oof, cat_oof])
meta_test = np.column_stack([lgb_test, xgb_test, cat_test])

# Meta-model
from sklearn.linear_model import LogisticRegression
meta = LogisticRegression(C=1.0)
meta.fit(meta_train, y_train)
final_preds = meta.predict_proba(meta_test)[:, 1]
```

## Achieving diversity

Ensembling gains scale with prediction correlation. Two models correlated at 0.99 hardly help each other; two models correlated at 0.85 combine much better. Sources of diversity:

- **Different algorithms.** LightGBM + XGBoost + CatBoost + Random Forest + Linear.
- **Different features.** Train one model on all features, another on just the numeric ones, another on just aggregations.
- **Different targets.** For a classification problem, train one model on the raw target, another on a rank-transformed target (regression on rank).
- **Different seeds.** Least valuable but free — adds slight diversity without any work.
- **Different objectives.** For regression, mixing MSE and MAE objectives creates useful diversity because they weight outliers differently.

Compute pairwise correlations of your OOF predictions before adding a model to the ensemble. If a new candidate correlates above 0.98 with an existing model, it likely doesn't contribute.

## Common failure modes

- **Meta-model overfits.** Cause: base model OOF predictions look better than test predictions (temporal shift, target leakage in base features, or too many folds relative to dataset size). Fix: use fewer folds, add strong regularization to the meta-model, or check for leakage in Level 0.
- **Blending weights unstable across submissions.** Cause: validation is too small to distinguish 0.35 from 0.40. Fix: use k-fold weight optimization instead of a single validation split.
- **Ensemble worse than best base model.** Cause: one base model is dominant and others hurt more than they help. Fix: try dropping the weakest model, or tune weights more carefully.
- **Massive train/test gap in stacked model.** Cause: meta-features from base models trained without proper OOF. Rebuild Level 0 with strict OOF discipline.

## Practical recipe for a competition finish

Two-week competition target:

- Week 1: build one strong LightGBM baseline with real feature engineering. Get to top-25% with this alone.
- Week 2 first half: add XGBoost and CatBoost trained on the same features. Simple average of the three. Push into top-15%.
- Week 2 second half: try a linear model on the same features (for diversity), tune blend weights. If time permits, one stacked meta-model. Push into top-10%.

Do not skip the strong baseline for early ensembling. An ensemble of five weak models is worse than one strong model.
