# CatBoost strategy for competition tabular data

## When CatBoost outperforms other gradient boosting libraries

CatBoost is often the winning choice when the dataset has many high-cardinality categorical features. Its ordered target statistics implementation avoids target leakage that plain target encoding introduces, which means categorical features get more useful signal without hand-tuned encoding pipelines.

Prefer CatBoost first when:

- Most predictive features are categorical (user_id, product_id, region, URL segments).
- Cardinality of key columns is high (>100 unique values each).
- You want a strong single-model baseline with minimal preprocessing.
- CPU training only — CatBoost's symmetric trees are unusually cache-friendly.

CatBoost is often the runner-up rather than the winner when the dataset is small (<5000 rows) or dominated by numeric features. LightGBM tends to edge it out on those.

## Baseline configuration

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    subsample=0.8,
    random_seed=42,
    eval_metric="AUC",              # match competition metric
    early_stopping_rounds=100,
    task_type="CPU",
    verbose=100,
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_column_indices,   # crucial — pass here, not preprocessed
)
```

Always pass `cat_features`. If you one-hot encode categoricals before fitting, you throw away CatBoost's biggest advantage.

## Hyperparameter priority order

1. **`depth`**. CatBoost uses symmetric trees, so depth directly controls capacity. Values 4-10; start at 6. Deeper than 10 usually overfits.
2. **`l2_leaf_reg`**. Regularization strength. Start at 3.0. Raise to 10+ when overfitting; lower to 1.0 when underfitting.
3. **`learning_rate` with `iterations`**. Same trade-off as LightGBM — halving learning rate and doubling iterations improves quality but slows training. Use early stopping to avoid wasting time.
4. **`bagging_temperature`** (or `subsample`). Bayesian bootstrap parameter. Higher values (1.0+) create more randomness between trees, useful when the model overfits after other regularization is exhausted.

## Native categorical handling — the details that matter

CatBoost's `cat_features` argument accepts either column indices or column names. What it does under the hood:

- For each categorical column, it computes ordered target statistics — the target mean for prior rows in a random permutation of the training set. This avoids the standard leakage of naive target encoding, where a value's own target contributes to its encoding.
- For splits, it considers combinations of categoricals (feature crosses) automatically up to a configurable depth.

Two important consequences:

1. **Do not pre-encode categoricals as integers with meaning.** CatBoost treats integers as numeric unless told otherwise; feed the raw string values and pass `cat_features`.
2. **The `simple_ctr` and `combinations_ctr` parameters control the ordered target statistics.** Defaults are strong; only tune these on huge cardinality columns where memory is tight.

## Text features

CatBoost supports text features natively via `text_features=`. It handles tokenization and TF-IDF-style transformation internally. This is a fast baseline for text-adjacent tabular problems (product descriptions, short comments). For a real NLP problem, use a transformer-based approach instead.

```python
model.fit(
    X_train, y_train,
    cat_features=cat_cols,
    text_features=text_cols,
    eval_set=(X_val, y_val),
)
```

## Common failure modes

- **Long training times on wide datasets.** Reduce `depth` first (each step down roughly halves training time), then reduce `iterations`.
- **Worse than LightGBM on numeric-heavy problems.** Expected. CatBoost's win region is high-cardinality categoricals; don't force it elsewhere.
- **Memory blowup with many categorical combinations.** Reduce `max_ctr_complexity` from the default 4 to 2 or 1.
- **CV score much better than leaderboard.** Check that your CV strategy respects any temporal or group structure. Ordered target statistics still leak if you cross a time boundary or group boundary within a fold.

## GPU training

`task_type="GPU"` typically speeds training 5-10x on medium-sized datasets. It requires CUDA. On Windows machines without a discrete GPU, stay on CPU — CatBoost's CPU implementation is unusually fast and often beats GPU on small data.
