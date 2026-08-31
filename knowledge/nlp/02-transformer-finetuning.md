# Transformer fine-tuning for NLP competitions

## When transformers earn their compute

A transformer fine-tune becomes worth the compute cost when at least one of:

- The competition metric rewards nuanced semantic understanding (long documents, paraphrase, entailment).
- The dataset is large enough for the model to learn meaningfully (>10k rows, ideally >100k).
- A TF-IDF baseline saturates well below the leaderboard top and you need a step change.
- Domain-specific pretraining is available (BioBERT for biomedical, LegalBERT for legal, FinBERT for finance).

Under 5k training rows, small transformers often perform worse than tuned TF-IDF baselines. Fine-tuning at that scale mostly overfits noise.

## Base model selection

| Dataset size | Compute | Base model | Rationale |
|---|---|---|---|
| < 5k rows | Any | Skip transformers; TF-IDF wins | Not enough signal to fine-tune. |
| 5k-20k | CPU or small GPU | DistilBERT | Faster fine-tune, similar quality to BERT on short texts. |
| 20k-200k | GPU (8GB+) | RoBERTa-base or DeBERTa-v3-base | Standard competitive choice. |
| 200k+ | GPU (16GB+) | DeBERTa-v3-large | Meaningful quality gains from larger model. |
| Multilingual | Any | XLM-RoBERTa or mBERT | Only choose these when multilingual coverage is needed. |
| Domain-specific | Any | Domain BERT variant | Meaningfully outperforms generic BERT on domain tasks. |

DeBERTa-v3 (either base or large) has been the strongest single-model choice in most 2023-2024 text competitions. Its disentangled attention mechanism handles positional signals better than earlier BERT variants.

## Fine-tuning hyperparameters that matter

The single most impactful parameter is learning rate. Get this wrong and no amount of other tuning helps.

```python
training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,               # 2-5 typical for competition fine-tunes
    per_device_train_batch_size=16,   # scale to your GPU
    per_device_eval_batch_size=64,
    learning_rate=2e-5,               # start here for BERT-family
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_auc",
    fp16=True,                        # roughly halves memory
    report_to="none",
    seed=42,
)
```

**Learning rate ranges by model:**

- DistilBERT, BERT-base: 2e-5 to 5e-5
- RoBERTa-base: 1e-5 to 3e-5
- DeBERTa-v3-base: 1e-5 to 3e-5
- DeBERTa-v3-large: 5e-6 to 1e-5 (large models need smaller LR)

If the training loss diverges to NaN, halve the learning rate. If loss barely moves after epoch 1, double it.

**Epochs:** More than 5 epochs on competition data almost always overfits. Watch validation loss and use early stopping.

**Batch size:** Larger is not always better. For BERT-family models on classification, batch sizes of 8-32 work well. Very small batches (2-4) with gradient accumulation are needed only on tiny GPUs.

**Warmup:** 10% of training steps is a safe default. Skipping warmup on transformer fine-tuning causes early training instability.

## Sequence length choices

The default 512-token max sequence length is expensive. Truncation strategy affects quality:

- **Short texts (<128 tokens on average):** set `max_length=128`. 4x faster than 512, no quality loss.
- **Medium texts (128-256 average):** use `max_length=256`. Truncate from the end.
- **Long documents (>512 tokens on average):** the model sees only the first 512 by default. Options:
  - Use a long-context model (Longformer, BigBird).
  - Chunk the document, embed each chunk, aggregate (mean-pool or attention-pool).
  - Truncate strategically: first 256 + last 256 tokens often carries more signal than the first 512.

## What almost always helps

- **Layer-wise learning rate decay.** Set a lower LR for early transformer layers than for the classification head. Common ratio: multiply LR by 0.9 per layer going deeper.
- **Multi-sample dropout.** Applies dropout multiple times to the same batch and averages outputs. Roughly 0.001-0.002 gain on binary classification, essentially free.
- **Adversarial training (FGM, AWP).** Adds small adversarial perturbations to embeddings during training. Consistent 0.001-0.003 gain on text classification; adds 30-50% training time.
- **Averaging checkpoints across epochs.** Store checkpoints from the last 3 epochs and average their weights. Simpler than ensembling, roughly as effective.

## What rarely helps

- **Longer pretraining before fine-tuning.** Unless you have a large domain corpus (millions of unlabeled examples) and days of compute, skip continued pretraining.
- **Custom tokenization.** The base tokenizer's vocabulary is fine for 99% of English tasks.
- **Complex classification heads.** A single linear layer on the CLS token performs as well as multi-layer heads for most classification problems.
- **Reducing dropout.** BERT-family default dropout (0.1) is well-tuned. Changing it rarely helps.

## Ensembling transformer fine-tunes

Fine-tune 3-5 models with different seeds and average their prediction probabilities. Typical gain: 0.002-0.005 over the best single seed.

For more diversity: mix architectures (RoBERTa + DeBERTa + XLNet), mix sequence lengths, or vary the classification head slightly. Diverse transformer ensembles regularly appear in top competition solutions.

## Common failure modes

- **Validation score much better than public leaderboard.** Typical cause: preprocessing computed statistics on the full dataset. Fix: apply preprocessing per-fold.
- **Loss goes to NaN in the first few steps.** Learning rate too high. Halve it. If still NaN with fp16, try fp32.
- **Fine-tuned model performs worse than TF-IDF baseline.** Check that (a) the label mapping is correct, (b) the learning rate is in the right range, and (c) the dataset actually rewards semantic understanding.
- **Out of memory on any batch size.** Reduce `max_length` before reducing `batch_size` (quadratic memory savings vs. linear).
