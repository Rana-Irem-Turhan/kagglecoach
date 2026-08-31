# Overfitting diagnosis framework

## The three signals of overfitting

Overfitting is when a model fits its training data more closely than the underlying pattern the training data samples. Three signals reveal it:

1. **Train/validation gap.** Training metric far better than validation metric.
2. **Validation instability.** Score varies wildly across CV folds or across small changes to the training procedure.
3. **Public/private leaderboard gap.** Public score much better than the private score revealed at competition end.

Signals 1 and 2 are what you can observe during a competition. Signal 3 is the ultimate check — a heavily overfit model that fits public leaderboard noise collapses when the private set is scored.

## Classifying gap severity

For a metric where higher is better (accuracy, AUC, F1):

| Gap | Classification | Response |
|---|---|---|
| < 3% relative | Healthy generalization | No action needed. |
| 3-8% relative | Mild overfitting | Standard regularization increase. |
| 8-15% relative | Moderate overfitting | Investigate features and model complexity. |
| > 15% relative | Severe overfitting | Stop and audit the pipeline before proceeding. |

For loss metrics (log loss, RMSE), the same buckets apply but reversed — validation loss above training loss by the same proportions.

## The audit sequence for suspected overfitting

Work through these in order. Skipping steps risks fixing the wrong problem.

### 1. Verify the gap is real

- Is the CV setup right for this data? Random splits on time-series or grouped data inflate CV artificially. Re-check against the validation strategy decision tree.
- Are you comparing apples to apples? Training metric computed with dropout enabled vs. validation with dropout disabled artificially inflates the gap.
- Is the validation set large enough to be trustworthy? Under 500 examples, a 5% gap is within noise.

If the setup is questionable, fix that before regularizing anything.

### 2. Look for leakage in features

Overfitting from feature leakage produces gaps that resist regularization because the leaky feature has real signal — just not signal that generalizes.

Look for:

- Features that are impossibly predictive on training. If a single feature scores over 0.95 AUC alone, it's likely a leak.
- Features computed with future information (rolling means, aggregations, target encoding without OOF folds).
- Post-target features (features derived from what you're trying to predict, or from downstream events).
- Features that identify individual rows (user_id used as a raw feature when the same users appear in test).

Verify: drop the suspect features one at a time and observe whether the gap shrinks. A shrinking gap identifies the leak.

### 3. Reduce model complexity

Once features are clean, address model capacity:

- **Tree ensembles:** raise `min_data_in_leaf` (LightGBM) or `min_child_weight` (XGBoost), lower `num_leaves` or `max_depth`, raise `l2_leaf_reg` or `reg_lambda`.
- **Neural networks:** add dropout (0.2-0.5), reduce hidden dim, add L2 weight decay (1e-4 to 1e-3), use early stopping on validation loss.
- **Linear models:** lower `C` (raise regularization strength).

Aim to close half the gap through complexity reduction. If the gap barely moves, the issue is not model complexity — return to feature auditing.

### 4. Add regularization-adjacent techniques

- **Dropout equivalents for tree models.** LightGBM's `bagging_fraction=0.7` and `feature_fraction=0.7` add stochasticity per tree.
- **Data augmentation.** For image and text tasks; for tabular, moderate feature noise or subsample-based training.
- **Ensembling.** Averaging across seeds reduces variance and often narrows the gap.
- **Longer training with lower learning rate.** For gradient boosting, halving learning rate and doubling rounds sometimes improves generalization at the cost of training time.

### 5. Reconsider dataset size

Fundamentally, some datasets are too small for the model class chosen. A large neural network on 1000 rows will overfit no matter how you regularize. Consider:

- Simpler model class (linear model, small tree ensemble).
- External data (competition rules permitting).
- Semi-supervised learning if unlabeled data exists.

## Diagnosing by dataset characteristics

Some overfitting risks are predictable from the data alone, before training anything:

- **Fewer than 50 rows per feature.** High risk. Feature selection or dimensionality reduction becomes essential.
- **Highly imbalanced classification (>95% majority).** Effective sample size for the minority class is what limits model complexity.
- **Small dataset with high-cardinality categorical.** Any target-encoded feature is a leakage risk. Use very heavy smoothing or drop the feature.
- **Time series with short history.** Under 1 year of daily data, most temporal features have too few instances to learn stable patterns.

## The gap-doesn't-close case

If overfitting persists after complexity reduction and feature auditing, three possibilities:

1. **The task is fundamentally hard given the data.** The validation score you're seeing might be close to the achievable ceiling. Compare against baselines and public leaderboard bests.
2. **CV configuration hides a real problem.** Try a different CV split (add a fold, change random seed) and see if the score is stable.
3. **The labels have noise.** Perfect fit to noisy labels means poor generalization by definition. Nothing beyond regularization helps this — you're at the label-noise ceiling.

Recognize case 1 or 3 and stop over-tuning. More epochs and more feature engineering do not overcome an achievable-ceiling situation.

## Common failure modes

- **Reducing complexity too aggressively.** A model that underfits produces low train and low validation scores. If the gap is small but both scores are poor, add back capacity rather than removing more.
- **Regularizing at the wrong layer.** Adding dropout on the input layer of a neural network usually hurts more than adding it on hidden layers.
- **Assuming test = validation.** In competitions with public/private leaderboard split, the public leaderboard is a small sample of the private one. Trust CV over public LB.
- **Selecting hyperparameters that "just barely fit."** Configurations at the edge of underfitting often score best on validation but are the most unstable. Prefer configurations 5-10% into the safe zone.
