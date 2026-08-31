# Evaluation metric implications for strategy

## The metric shapes every subsequent decision

The evaluation metric is not just how the leaderboard scores you — it dictates which model families to consider, which features to prioritize, how to handle class imbalance, whether to calibrate probabilities, and how to combine models in an ensemble. Reading the metric carefully in the first hour of a competition saves days of wasted effort later.

## Classification metrics

### Log loss (binary cross-entropy)

Log loss rewards well-calibrated probabilities. Confident wrong predictions are penalized heavily — predicting 0.99 for a negative example costs far more than predicting 0.6.

**Strategy implications:**

- Use models that output probabilities natively (logistic regression, calibrated tree ensembles).
- Do not tune a decision threshold — log loss uses the raw probability.
- Calibrate your model with `CalibratedClassifierCV` if it's poorly calibrated (tree ensembles often are).
- Ensembling by averaging probabilities usually improves log loss more than accuracy-based metrics.

**Common mistakes:**

- Clipping predictions too aggressively (e.g., to `[0.05, 0.95]`) hurts calibrated models. Clip only if the model produces exactly 0 or 1.
- Using accuracy as a proxy — a model with the same accuracy but better calibration wins big on log loss.

### ROC AUC

AUC measures how well the model ranks positives above negatives. It's threshold-independent — you don't need to pick a decision boundary.

**Strategy implications:**

- Only the ordering of predictions matters. Absolute probabilities are irrelevant. Calibration is unnecessary.
- Class imbalance doesn't shift AUC (unlike accuracy or F1).
- Ensembling by averaging ranks (not probabilities) often works better than averaging predictions.
- Feature engineering that improves ranking wins; feature engineering that just shifts probabilities doesn't help.

**Where AUC can mislead:**

- AUC weights performance across all thresholds equally. If the competition rewards a specific operating point (say, precision at 90% recall), AUC may not track that region.
- On extreme imbalance (<1% positive), AUC can be high while precision at reasonable recall is terrible. PR AUC is more honest there.

### PR AUC / Average Precision

Precision-Recall AUC focuses on the positive class and is much less generous than ROC AUC on rare-positive problems.

**When to use:**

- Rare-event competitions (fraud, medical diagnosis of rare conditions, anomaly detection).
- Any classification problem where the majority class isn't the focus.

**Strategy implications:**

- Similar to ROC AUC — ranking matters, not calibration.
- Optimizing for PR AUC often pushes models to be more aggressive on positive predictions than log-loss-optimized models.
- Class weighting toward the positive class often helps directly.

### F1 (and F-beta variants)

F1 is the harmonic mean of precision and recall at a fixed threshold. F-beta variants let you weight one over the other.

**Strategy implications:**

- Threshold tuning is critical. The default 0.5 is almost never optimal on imbalanced problems.
- Class weighting during training helps; threshold tuning after training helps more.
- Ensembling improves F1 less directly than log loss or AUC — you need to also re-tune the threshold on the ensemble.

### Accuracy

The simplest metric and the most misleading on imbalanced data.

**Strategy implications:**

- On balanced datasets (roughly 40-60% positive), accuracy is a fine proxy for AUC.
- On imbalanced datasets, accuracy rewards predicting the majority class. Ignore it as a training signal.
- Threshold tuning matters if you're forced to submit predictions rather than probabilities.

### Multi-class metrics

`accuracy` scales naturally to multi-class but shares the imbalance issue.

`log_loss` extends directly (categorical cross-entropy).

`f1_score(average=...)` has three meaningful modes:

- **macro:** per-class F1 averaged unweighted. Rewards rare-class performance.
- **weighted:** per-class F1 weighted by class support. Larger classes dominate.
- **micro:** aggregate globally. Equivalent to accuracy for multi-class classification.

Check which averaging the competition uses — the strategic implications differ.

## Regression metrics

### RMSE (Root Mean Squared Error)

Squares the errors, so outliers dominate the loss.

**Strategy implications:**

- The model spends its capacity fitting the largest errors. Outlier handling in preprocessing matters.
- Log-transforming the target often helps when the target is right-skewed and has occasional large values.
- Ensembling by simple averaging directly reduces variance in the RMSE sense.

### MAE (Mean Absolute Error)

Linear in error magnitude. Robust to outliers.

**Strategy implications:**

- Model spends capacity on the typical case, not the outliers.
- Gradient boosting with the MAE objective (`L1` loss) trains more slowly than MSE. Some libraries approximate it.
- Median of predictions (not mean) is the optimal aggregation for MAE. Use it when ensembling.

### RMSLE (Root Mean Squared Logarithmic Error)

RMSE on the log-transformed target. Penalizes relative error rather than absolute.

**Strategy implications:**

- Train on `log1p(target)` and predict `expm1(prediction)` at inference. RMSE on the transformed target directly optimizes RMSLE.
- Common for problems with wide-ranging targets (prices, counts, populations).

### MAPE (Mean Absolute Percentage Error)

Percentage error. Undefined when true values are zero; unstable near zero.

**Strategy implications:**

- Not directly optimized by standard loss functions. Approximate by training on log-transformed target with MSE loss.
- Consider whether the competition uses SMAPE (symmetric MAPE) instead — the numerical properties differ.

### R² (coefficient of determination)

Fraction of variance explained. Negative for models worse than predicting the mean.

Rarely used as a competition metric because it's just a monotonic transformation of RMSE relative to the baseline. Interpret it as diagnostic, not as an optimization target.

## Ranking metrics

### NDCG (Normalized Discounted Cumulative Gain)

Standard for search and recommendation. Rewards putting relevant items at the top of a ranked list, with position-dependent discounts.

**Strategy implications:**

- Train models with pairwise or listwise ranking objectives (LambdaMART, LightGBM's `lambdarank`).
- Regression models predicting relevance scores work but usually underperform ranking-native objectives.
- Cutoff variants (NDCG@10, NDCG@100) reward slightly different behaviors — the shorter the cutoff, the more precision at the top matters.

### MAP (Mean Average Precision)

Common for retrieval competitions. Similar strategic implications to NDCG.

## The metric-to-strategy summary table

| Metric | Model type | Threshold tuning | Calibration | Ensemble style |
|---|---|---|---|---|
| Log loss | Probabilistic | No | Yes | Average probabilities |
| ROC AUC | Any | No | No | Average ranks or probabilities |
| PR AUC | Any (rare-positive) | No | No | Average ranks or probabilities |
| F1 | Classifier | Yes | Slightly | Average probs, then re-tune threshold |
| Accuracy | Classifier | Yes | No | Vote |
| RMSE | Regression | N/A | N/A | Simple average |
| MAE | Regression | N/A | N/A | Median |
| RMSLE | Regression | N/A | N/A | Log-space average |
| NDCG | Ranker | N/A | N/A | Reciprocal rank fusion |

Use this table as a first sanity check when starting a competition.
