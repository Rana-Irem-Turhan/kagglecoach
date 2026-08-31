# XGBoost strategy for competition tabular data

## When XGBoost is the right choice

XGBoost predates both LightGBM and CatBoost and remains competitive when:

- The dataset is medium-sized (10k-1M rows) with a mix of numeric and low-cardinality categoricals.
- You need highly stable training — XGBoost's deterministic algorithm produces more consistent results across seeds than LightGBM's histogram-based sampling.
- You're building an ensemble and want algorithmic diversity — a well-tuned XGBoost + LightGBM average often beats either alone.
- The submission environment has a mature XGBoost setup (many Kaggle grandmaster teams stick with it for pipeline reasons).

For most fresh single-model submissions in 2024+, LightGBM matches or exceeds XGBoost quality while training 2-3x faster. Pick XGBoost when you have a reason, not by default.

## Baseline configuration

```python
import xgboost as xgb

params = {
    "objective": "binary:logistic",     # or "multi:softprob" / "reg:squarederror"
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",              # much faster than "exact"
    "n_jobs": -1,
    "seed": 42,
}

model = xgb.XGBClassifier(**params, n_estimators=5000, early_stopping_rounds=100)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
```

The `tree_method="hist"` setting is essential — it uses the same histogram-based algorithm that makes LightGBM fast. Legacy XGBoost defaults are much slower.

## Hyperparameter priority order

1. **`max_depth`**. Controls tree complexity. Unlike LightGBM's `num_leaves`, this is a hard depth cap. Values 3-10; start at 6. Deeper than 10 rarely helps and typically overfits.
2. **`min_child_weight`**. Minimum sum of instance weights needed in a leaf. Regularization lever. Raise to 20-100 for small datasets, keep at 1-5 for large ones.
3. **`learning_rate` paired with `n_estimators`**. Same trade-off as elsewhere. Use early stopping.
4. **`subsample` and `colsample_bytree`**. Both in [0.5, 1.0]. Add stochasticity to reduce overfitting; drop from 0.8 to 0.6 when the model overfits despite depth/min_child_weight tuning.
5. **`reg_alpha` and `reg_lambda`**. L1 and L2 regularization on leaf weights. Rarely the biggest lever; tune late.

## Categorical handling

XGBoost added native categorical handling in v1.5+ via `enable_categorical=True` and pandas `category` dtype. It works but is less mature than LightGBM's or CatBoost's implementation. For competitions:

- Low cardinality (<10 unique): one-hot encoding is fine.
- Medium cardinality (10-50): use the native categorical handling.
- High cardinality (>50): use target encoding (with proper out-of-fold folds) or move to CatBoost.

## Common failure modes

- **Massively slower than LightGBM.** Confirm `tree_method="hist"`. Without it, XGBoost uses exact splits and takes 5-10x longer.
- **Divergent CV vs leaderboard.** Same causes as LightGBM — leakage in preprocessing, target contamination, temporal or group violations in the CV split.
- **Loss stuck at random baseline.** Usually a label encoding issue — check that binary targets are {0, 1} and multiclass targets are contiguous integers starting from 0.
- **Out of memory on wide datasets.** Reduce `colsample_bytree` to 0.5 first; then consider `tree_method="approx"` (slower but memory-efficient).

## GPU acceleration

Set `device="cuda"` (v2.0+) or `tree_method="gpu_hist"` (older versions) to train on GPU. Typical speedup is 3-8x on medium-to-large datasets. For datasets under 100k rows, GPU overhead often makes CPU training equal or faster.

## Ensembling

XGBoost ensembles well with LightGBM. Train both on the same features, average their predictions (or blend with logistic regression on validation predictions), and the ensemble typically beats the stronger single model by 0.001-0.005 on AUC-style metrics.

For maximum diversity in an ensemble, vary the objective slightly (e.g., logistic vs. hinge for classification), the base tree structure (`booster="gbtree"` vs. `"dart"`), and the seed. Dropout-style `dart` boosting often trains slower but produces meaningfully different predictions.
