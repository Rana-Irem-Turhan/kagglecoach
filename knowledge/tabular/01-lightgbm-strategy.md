# LightGBM strategy for competition tabular data

## When LightGBM is the right first pick

LightGBM is the default first model for most tabular competitions on datasets between 10k and 10M rows. It handles mixed numeric and categorical features natively, deals with missing values without imputation, and trains 3-5x faster than XGBoost at comparable quality on most problems.

Reach for it first when:

- The target is well-defined (classification with clear labels, or regression with a bounded target).
- There are more numeric features than high-cardinality categoricals.
- You have less than a day of compute for the entire modeling loop.
- You need to iterate on features quickly — LightGBM's fast training makes feature engineering experiments cheap.

Consider CatBoost first instead when the dataset is dominated by high-cardinality categoricals (user_id, product_id, url) — CatBoost's ordered target encoding often wins on those without leakage risk.

## Baseline hyperparameters that just work

Start with these and only tune after you have a real feature engineering pipeline. Tuning before features usually improves the wrong axis.

```python
params = {
    "objective": "binary",              # or "multiclass" / "regression"
    "metric": "auc",                    # match the competition metric
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,                    # no limit; num_leaves controls
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}
```

Train with early stopping. Set `num_boost_round` to something generous (5000-10000) and let `early_stopping_rounds=100` decide when to stop.

## Hyperparameter priority order

If you only have time to tune three things, tune these in order:

1. **`num_leaves`**. The single most impactful parameter. Values from 15 (heavy regularization) to 255 (high capacity). Rule of thumb: start at 63, go down if overfitting, up if underfitting. Never exceed `2^max_depth - 1` when both are set.
2. **`min_data_in_leaf`**. Second lever against overfitting. Small datasets: 100-500. Large datasets: 20-100. Interacts strongly with `num_leaves` — raise both together if the model overfits.
3. **`learning_rate` paired with `num_boost_round`**. Halving the learning rate and doubling the round count often improves generalization at the cost of training time. Do this only in the final push before submission.

Ignore `max_depth` unless you know why. `num_leaves` is the more direct control on tree complexity.

## Categorical handling

Pass categorical columns as pandas `category` dtype or specify their column indices in `categorical_feature=`. LightGBM's built-in categorical split beats one-hot encoding for cardinalities above 8-10 categories.

Never one-hot encode a high-cardinality column (>50 unique values) for LightGBM. It fragments the split search and hurts quality. Either use the native categorical handling or apply target encoding with proper out-of-fold computation.

## Common failure modes

- **Overfitting on small datasets (<5k rows).** Raise `min_data_in_leaf` to 100+, lower `num_leaves` to 15-31, and consider whether a simpler model (linear + regularization) is actually the right choice.
- **Underperforming on high-cardinality categoricals.** Try target encoding (with 5-fold OOF) before falling back to the native categorical handling.
- **Slow training.** Set `feature_fraction` lower (0.5-0.6) rather than reducing `num_leaves` — you keep model capacity while cutting per-tree cost.
- **Metric gap between local CV and public leaderboard.** Usually indicates leakage in your feature engineering pipeline. Audit any features derived from the target, from time-ordered data, or from grouped rows.

## Ensembling within LightGBM

For a competition-grade single-model submission, train 5 LightGBM models with different seeds and average their predictions. This alone typically buys 0.001-0.003 on AUC-style metrics — worth it if you're near the leaderboard cutoff.

For more diversity, vary two additional hyperparameters between the seeds (e.g., num_leaves in {31, 63, 127} and feature_fraction in {0.7, 0.8, 0.9}). The correlation between predictions drops and the average improves.
