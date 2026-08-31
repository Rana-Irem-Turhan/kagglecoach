# Handling class imbalance and pseudo-labeling in NLP tasks

## Diagnosing class imbalance in text classification

Class imbalance shows up in most NLP competitions — abuse detection, medical diagnosis, low-resource classification. Two thresholds trigger different responses:

- **Mild imbalance (majority class 60-80%).** Standard training usually works. Watch precision/recall separately and use F1 or PR AUC instead of accuracy.
- **Moderate imbalance (majority 80-95%).** Class weighting and threshold tuning become essential.
- **Severe imbalance (majority >95%).** Requires deliberate strategy — augmentation, sampling, or specialized losses.

Determine which regime you're in by checking `y_train.value_counts(normalize=True)` before training anything.

## Strategies by imbalance severity

### Mild (60-80%)

Do nothing special beyond using the right metric. Report and monitor precision, recall, and F1 rather than accuracy. Consider `class_weight="balanced"` on the classifier if the minority class is what matters.

### Moderate (80-95%)

Combine two or three of:

- **Class weighting.** For sklearn models, `class_weight="balanced"`. For neural training, pass `weight=` to `CrossEntropyLoss`.
- **Focal loss.** `FL(pt) = -(1-pt)^gamma * log(pt)`. Down-weights easy examples so the model focuses on hard cases. `gamma=2` is the standard starting point.
- **Threshold tuning.** Do not use the default 0.5 threshold on imbalanced problems. Sweep thresholds on validation and pick the one that maximizes your metric.

```python
from sklearn.metrics import f1_score
import numpy as np

thresholds = np.linspace(0.05, 0.95, 91)
best_f1, best_t = 0.0, 0.5
for t in thresholds:
    f1 = f1_score(y_val, val_probs >= t)
    if f1 > best_f1:
        best_f1, best_t = f1, t
```

### Severe (>95%)

Additional techniques:

- **Under-sampling the majority.** Randomly drop majority examples until the ratio is 80-90/20-10. Combined with class weighting, often the strongest quick fix.
- **Oversampling with augmentation.** Duplicate minority examples with variations (see augmentation section below).
- **Back-translation.** Translate minority examples through another language and back. Adds paraphrased versions of the minority class.
- **Two-stage training.** First train on a balanced subsample to learn class separability; then continue on the full imbalanced set to calibrate probabilities.

Do not use SMOTE on text features. It interpolates in embedding space and produces nonsensical text-adjacent artifacts.

## Text augmentation techniques

Effective augmentation preserves the label while changing surface form:

- **Back-translation.** English → German → English produces paraphrases that retain meaning. Fluent and label-preserving; the strongest general-purpose text augmentation.
- **Synonym replacement.** Replace a small fraction of words (10-15%) with WordNet synonyms. Cheap but often hurts on domain text where synonyms shift meaning.
- **Random insertion / swap / deletion.** Simple operations from the EDA (Easy Data Augmentation) paper. Fast, applied at training time, roughly 0.001-0.005 gain on small imbalanced datasets.
- **Contextual augmentation.** Use a masked language model to replace tokens with predictions in context. Better label preservation than synonym replacement.
- **Model-based paraphrase.** Feed examples through T5 or a paraphrasing model. Slow but produces high-quality varied augmentations.

Only augment the minority class, and only during training (never during evaluation). Augmenting both classes equally does not address imbalance.

## Pseudo-labeling: when and how

Pseudo-labeling uses your model's confident predictions on unlabeled data (usually the test set) as pseudo-training-labels. Effective when a large unlabeled pool exists and your model has room to learn from more data.

**The safe recipe:**

1. Train the base model on labeled training data.
2. Predict on unlabeled data (test or external corpus).
3. Keep predictions where the model is highly confident (say, `max_prob > 0.9`).
4. Add these confident examples with their pseudo-labels to the training set.
5. Retrain the model on the combined data.
6. Optionally repeat once with the retrained model's higher-confidence predictions.

**The risks:**

- **Confidence miscalibration.** If the base model's high-confidence predictions are systematically wrong (e.g., biased toward the majority class), pseudo-labeling amplifies the bias.
- **Distribution shift.** If the test set differs from the train set in ways your model can't detect, pseudo-labels reinforce test-set biases.
- **Feedback loops.** After several rounds, pseudo-labels dominate real labels and the model's errors compound.

Guardrails: cap the pseudo-labeled dataset at 2-3x the labeled dataset size, keep the confidence threshold high (0.9+), and always evaluate on held-out labeled validation data (never on pseudo-labels).

## Multi-task and auxiliary training

When labeled data is scarce, adding related auxiliary tasks helps the model learn better representations:

- **Related labels.** If the competition target is "toxic vs. not toxic," also predict finer sub-categories (insult, threat, obscenity) as auxiliary heads. Sub-category signals often generalize.
- **Self-supervised objectives.** Masked language modeling on the competition's unlabeled data before fine-tuning. Adds a domain-adaptation step at the cost of extra compute.
- **Adversarial domain adaptation.** For train/test domain shift, add a discriminator that tries to distinguish train vs. test examples. The main model's features are then made domain-invariant.

Auxiliary tasks help most when labeled data is small (<10k) and the auxiliary signals are cheap to obtain.

## Common failure modes

- **F1 jumps massively but AUC barely moves.** You tuned the threshold; the ranking didn't improve. Fine at submission time, but don't confuse this with model improvement.
- **Pseudo-labeling improves CV but hurts leaderboard.** The base model's confident predictions are biased in a way that CV doesn't detect. Reduce the pseudo-labeled fraction or raise the confidence threshold.
- **Class weighting causes training instability.** For severely imbalanced problems (>99% majority), very large class weights make gradients explode. Combine with focal loss or under-sampling instead of relying on weights alone.
- **Augmentation hurts more than helps.** Usually because augmented text is too different from real text. Reduce augmentation strength or switch to back-translation.
