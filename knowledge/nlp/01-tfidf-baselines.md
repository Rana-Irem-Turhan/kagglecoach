# TF-IDF baselines for NLP competitions

## Why start with TF-IDF

Every NLP competition should begin with a TF-IDF baseline before any deep learning attempt. Reasons:

- **Sets a real floor.** If your fancy transformer doesn't beat a well-tuned TF-IDF baseline by a meaningful margin, something is wrong with your fine-tuning setup.
- **Reveals data properties.** The features TF-IDF finds important (via linear model coefficients) tell you which words carry signal, informing later modeling choices.
- **Ships fast.** A tuned TF-IDF + Logistic Regression pipeline trains in seconds on datasets under 100k rows.
- **Stays competitive on short-text problems.** For classification of sentences, short reviews, or single-clause labels, TF-IDF pipelines regularly finish in the top 30% of competitions.

## The canonical baseline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )),
    ("clf", LogisticRegression(
        C=1.0,
        max_iter=1000,
        n_jobs=-1,
        solver="liblinear",
    )),
])
pipe.fit(train_text, y_train)
val_preds = pipe.predict_proba(val_text)[:, 1]
```

This alone often reaches within 3-8% of state-of-the-art on typical text classification benchmarks.

## Parameter priority for TF-IDF

Tune these in order of impact:

1. **`ngram_range`**. `(1, 1)` for very short texts (single sentences), `(1, 2)` for medium documents, `(1, 3)` when vocabulary is medium and dataset is large enough to support it (>50k rows). Higher ngrams grow the vocabulary quickly; watch memory.
2. **`min_df`**. Filters rare tokens. `2` or `3` is standard; too low includes noise, too high loses signal.
3. **`max_df`**. Filters overly common tokens. `0.95` is standard for topical classification; `1.0` (disable) sometimes helps when stop-word-like tokens carry signal.
4. **`sublinear_tf`**. `True` almost always improves logistic regression's performance on longer texts.
5. **`analyzer="char_wb"` with `ngram_range=(3, 5)`**. Character n-grams. Useful for multilingual text, misspellings, and code-mixed content. Combine with word n-grams via a `FeatureUnion` when both help.

## Classifier choice on TF-IDF features

- **Logistic Regression.** Default. Interpretable coefficients, calibrated probabilities, handles high-dimensional sparse features well. Solver `"liblinear"` for L1 or `"lbfgs"` for L2.
- **Linear SVM.** Slightly stronger than LR on some text tasks. No native probabilities; use `CalibratedClassifierCV` if you need them.
- **Naive Bayes.** Multinomial NB is the classic pairing with TF-IDF. Weaker than LR on most modern datasets but blends well in ensembles because errors are uncorrelated.
- **LightGBM on TF-IDF.** Rarely wins on raw text but useful in stacked ensembles where you want a non-linear view of the same features.

## Regularization on Logistic Regression

The `C` parameter is inverse regularization strength. Sweep it on a log scale:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"clf__C": [0.01, 0.1, 0.5, 1.0, 3.0, 10.0]}
search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
search.fit(train_text, y_train)
```

For most text classification tasks the optimal `C` is between 0.1 and 3.0. Values above 10 are rare and usually indicate overfitting.

## Preprocessing decisions

The TF-IDF vectorizer's built-in preprocessing (lowercasing, accent stripping) handles most needs. Extra preprocessing helps in specific cases:

- **Domain-specific tokens.** URLs, mentions, hashtags. Either strip them or replace with placeholder tokens (URL, USER, TAG) before vectorization.
- **HTML.** Strip tags with `BeautifulSoup` or a regex before feeding text to the vectorizer.
- **Emojis and non-ASCII.** Modern datasets carry signal in emojis; don't strip them blindly.
- **Very short texts.** Consider adding character n-grams to capture morphological signal that word n-grams miss.

Do not lemmatize or stem by default. On modern datasets the gain is small and can hurt when domain vocabulary is idiosyncratic. Test both if time permits.

## What TF-IDF cannot capture

TF-IDF represents documents as bags of tokens. It misses:

- **Word order beyond n-grams.** Long-range dependencies (question-answer patterns, negation scope) are invisible.
- **Semantic similarity.** "car" and "automobile" are unrelated to TF-IDF unless they co-occur with the same neighbors.
- **Context-dependent meaning.** "bank" as a river bank vs. financial institution is one feature.

These are the exact places transformer-based models beat TF-IDF. If your competition rewards them (long documents, semantically rich labels, cross-lingual tasks), plan the transformer path early.

## Common failure modes

- **Vocabulary too large, out of memory.** Reduce `ngram_range`, raise `min_df`, or use `HashingVectorizer` instead (fixed vocabulary size, no dictionary).
- **Poor performance on imbalanced targets.** Use `class_weight="balanced"` on the classifier, not resampling.
- **CV much better than leaderboard.** Usually indicates target leakage in the text itself (dates, IDs, publication metadata embedded in the raw text). Strip such fields.
- **Character n-grams blow up training time.** They generate an order of magnitude more features than word n-grams. Use only when word n-grams underperform.
