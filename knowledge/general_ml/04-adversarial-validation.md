# Adversarial validation and train/test distribution shift

## When to suspect distribution shift

Distribution shift between training and test data is a silent killer of competition submissions. It shows up as:

- A CV score that keeps improving while the public leaderboard stagnates or drops.
- Different models winning locally versus on the leaderboard.
- Feature importance that changes dramatically when you subsample training data.
- A gap between the training data's summary statistics and the test data's summary statistics.

Suspect shift whenever the competition combines data from different time periods, geographic regions, source systems, or collection methodologies. Read the competition description for hints — organizers often disclose that "test data is from a later time period" or similar.

## The core technique: adversarial validation

Adversarial validation quantifies how different two datasets are by training a classifier to distinguish them.

**Procedure:**

1. Take your training features `X_train` and your test features `X_test`.
2. Concatenate them; label training rows `is_test=0` and test rows `is_test=1`. This is your adversarial target.
3. Train a binary classifier (LightGBM works well) to predict `is_test` from features. Do not use the competition's original target.
4. Measure the classifier's AUC via cross-validation.

**Interpretation of AUC:**

| Adversarial AUC | Interpretation |
|---|---|
| < 0.55 | Distributions are near-identical. Standard CV is trustworthy. |
| 0.55 - 0.70 | Mild shift. Watch the CV/leaderboard gap; consider using distribution-aware sample weighting. |
| 0.70 - 0.85 | Substantial shift. Adjust training strategy — use predicted `is_test` probabilities to weight training samples toward test distribution. |
| > 0.85 | Severe shift. The training set is only marginally representative of test. Deeper intervention required (see below). |

## Using adversarial validation to identify shift-causing features

The classifier's feature importance tells you exactly which features differ between train and test:

```python
import lightgbm as lgb
import pandas as pd

X_combined = pd.concat([X_train.assign(is_test=0),
                        X_test.assign(is_test=1)]).reset_index(drop=True)
y_adv = X_combined.pop("is_test")

model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31)
model.fit(X_combined, y_adv)

importance = pd.Series(model.feature_importances_,
                       index=X_combined.columns).sort_values(ascending=False)
print(importance.head(20))
```

Any single feature that dominates importance for the adversarial classifier is a shift signal. Common culprits:

- **Time-derived features** (dates, timestamps, cyclic features from time). Usually differ by design if train and test come from different periods.
- **Row identifiers** (ID columns, sequence numbers). Drop these before adversarial validation — they're trivially distinguishing but carry no signal.
- **Metadata** (source_system, batch_id, version). If train comes from one source and test from another, these will dominate.

Once you identify shift-causing features, three options:

1. **Drop them.** Cleanest fix if the features don't carry real predictive signal.
2. **Transform them.** For time features, replace absolute values with relative differences (days since first observation).
3. **Downweight them.** Reduce their influence via feature-level weighting or removal from the model.

## Sample weighting toward test distribution

When distributions differ but every feature is legitimately useful, weight training samples by their probability of belonging to the test distribution:

```python
# Predict is_test probability for each training row using the adversarial classifier
train_test_probs = model.predict_proba(X_train)[:, 1]

# Use these as sample weights during the main model training
main_model.fit(X_train, y_train, sample_weight=train_test_probs)
```

This upweights training rows that look most like test data. It's a mild but often effective way to bridge the gap.

**Caveat:** if adversarial AUC is very high, the weights become extreme (some samples near 0, some near 1). Clip weights to a range like [0.1, 3.0] to avoid over-weighting a small subset.

## Selecting a training subsample similar to test

For severe shift, subsampling can outperform weighting. Rank training rows by their adversarial `predict_proba` for `is_test`; keep the top 30-70% most test-like rows:

```python
threshold = np.percentile(train_test_probs, 50)   # keep top 50%
mask = train_test_probs > threshold
X_train_test_like = X_train[mask]
y_train_test_like = y_train[mask]
```

Trade-off: fewer training rows means the main model has less data. Only beneficial when the discarded rows are actively misleading.

## Adversarial validation as a validation strategy

Beyond diagnosis, adversarial validation can construct better validation splits. Instead of random CV, split training data so that the validation fold is composed of the rows most similar to test:

1. Compute `is_test` probability for each training row (as above).
2. Sort training rows by decreasing `is_test` probability.
3. Take the top N rows (matching typical test-set size) as your validation set.
4. Train on the rest.

This gives a validation set that better approximates the true test-set difficulty. Especially useful for time-forward competitions where the most recent training rows are the closest analogue to test.

## What adversarial validation cannot fix

- **Target distribution shift.** If test targets follow a different distribution than train targets (label shift), no feature-based intervention helps. You need domain adaptation techniques on the labels, which usually require some labeled test-like data.
- **Concept drift.** When the relationship between features and target changes between train and test. Adversarial validation can identify feature distribution shift but not shift in the target's conditional distribution.
- **Small test set.** Adversarial validation needs enough test rows to train a classifier meaningfully. Under a few hundred test rows, the signal is unreliable.

## Combining with other techniques

Adversarial validation composes well with other tools:

- **After adversarial subsampling, run standard CV on the subsample.** The subsample better approximates test distribution.
- **Weight-based training with StratifiedKFold.** Preserve class balance while shifting distribution toward test.
- **Adversarial-weighted ensemble.** Weight each base model's contribution to the ensemble by its adversarial validation score, favoring models that look most like they'd generalize.

## Common failure modes

- **Adversarial AUC of 1.0.** Something is trivially distinguishing — usually an identifier or timestamp column left in the feature set. Remove and retry.
- **Adversarial weights make training unstable.** Cap weights and reduce learning rate.
- **CV improvement doesn't translate to leaderboard.** Distribution shift may not be the main issue. Recheck feature leakage and CV setup first.
- **Test set is too small to run adversarial validation reliably.** Skip and rely on domain knowledge about likely shift patterns.
