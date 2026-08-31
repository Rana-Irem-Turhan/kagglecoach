# KaggleCoach design

## The core design decision: hybrid rule-based + LLM

KaggleCoach's central design choice is that **the reasoning that must be reliable is deterministic Python code, and the LLM is used only for what it's actually good at**. The alternative — a single LLM prompt that receives a dataset and dialogue answers and generates a full strategy report — was rejected up front, for reasons this document lays out.

### What the LLM does well

Given a well-formed prompt with retrieved evidence, a small local model (phi-3.5-mini class, ~3.8B parameters) can:

- Summarise a set of retrieved chunks into readable prose.
- Apply the constraints in a user's dialogue context to a recommendation.
- Cite source filenames when instructed to.

### What the LLM does poorly at this scale

- **Multi-step statistical reasoning about a specific dataset.** Given "500 rows × 100 features, imbalance 95/5, one high-cardinality categorical", asking the model to produce a validation strategy, an overfitting risk assessment, and a hyperparameter starting point requires it to combine four independent decision trees. Small models fail this consistently — they mix strategies, invent thresholds, or hallucinate specific numbers.
- **Adaptive dialogue.** Asking the model "given these prior answers, what should the next question be?" produces incoherent question flow at this parameter count. The model routinely repeats questions, asks orthogonal ones, or terminates prematurely.
- **Specific numerical estimates.** "Expected improvement" numbers, "baseline score ranges", and "hyperparameter starting values that will work for this specific data" are outside what a 3.8B model can produce reliably. When it does produce them, they're often plausibly-wrong — which is worse than obviously-wrong because it slips through review.

### The split KaggleCoach uses

| Component | Approach | Why |
|---|---|---|
| Dialogue (5-7 branching questions) | Deterministic state machine (`dialogue.py`) | Guaranteed coverage, guaranteed termination, testable. |
| Dataset profile (EDA) | Pure pandas (`eda.py`) | Statistics are what pandas exists for. No LLM needed. |
| Overfitting risk assessment | Rule-based combination table (`overfitting.py`) | Combining "rows per feature" + "train/val gap" is 5 lines of if/else — deterministic, matches published heuristics. |
| Validation strategy selection | Rule-based decision tree (`validation.py`) | The tree is short and well-established. Encoding it as code makes it inspectable. |
| Model selection reasoning | LLM + retrieved evidence | Requires reading how the knowledge base characterises trade-offs; LLM's strength. |
| Feature engineering suggestions | LLM + retrieved evidence | Same — trade-off narrative from evidence. |
| Report rendering | Template (`report.py`) | Every number and section comes from the deterministic layers above; template just formats. |

The result is a report where numbers, thresholds, strategy names, and rationale citations trace back to specific code paths or specific source files. Nothing is invented at render time.

## Three-collection retrieval

The knowledge base is split across three FAISS collections rather than one:

- `tabular` (5 documents): LightGBM/CatBoost/XGBoost, feature engineering, ensembles.
- `nlp` (3 documents): TF-IDF baselines, transformer fine-tuning, imbalance & pseudo-labelling.
- `general_ml` (4 documents): validation strategies, evaluation metrics, overfitting diagnosis, adversarial validation.

### Why three collections instead of one

Three considerations pushed the design here:

1. **Query routing.** The dialogue's first answer determines the task family (tabular / NLP / other). Retrieving from the wrong domain injects noise. Routing to a primary collection based on that first answer keeps evidence focused.
2. **Confidence gating per collection.** The primary collection has a `min_strong_hits_primary` threshold (2 by default) below which the report notes reduced confidence. Supporting collections don't get this gate — they contribute cross-cutting concerns (validation, metrics) that apply regardless of task family. Having them in one collection would make the "primary strong hits" check meaningless.
3. **Scaling.** As the knowledge base grows, adding a new domain (image, RL, time series) is a matter of adding a folder and a line in `config.toml`. One-collection designs eventually need re-chunking or reorganisation.

### FAISS at this scale

The three collections combined are under a thousand chunks — a numpy matrix multiplication would work fine. FAISS is used because the spec calls for it, and because it makes the "add a new collection" path obvious: initialise another `IndexFlatIP`, dump it as a file, register it. When the corpus grows past hundreds of thousands of chunks, the same code seamlessly moves from `IndexFlatIP` to `IndexIVFFlat` or `IndexHNSW` without changing the surrounding logic.

Vectors are L2-normalised before insertion. `IndexFlatIP` on normalised vectors returns cosine similarity — same math as a naive `X @ q.T`, just wrapped in FAISS's search API.

## Foundry Local + optional Azure OpenAI

Embeddings **always** stay on Foundry Local. The uploaded CSV's summary statistics feed into the LLM prompt, but the raw data never leaves the machine. This matters because in most competition scenarios the dataset is either private (regulated data, proprietary features) or subject to terms of use that forbid external transmission.

Chat generation defaults to Foundry Local. A sidebar toggle switches chat to Azure OpenAI when the user explicitly opts in (and has env vars set). Two things about this design:

- **The switch is user-driven, not automatic.** Automatic coherence-based fallback was rejected — coherence detection is unreliable at this scale, and silently sending prompts to an external service after a "quality check" fails is a bad user experience for a locally-oriented tool.
- **Embeddings never move.** Only chat prompts do. This preserves the dataset-privacy guarantee even in Azure mode.

## Rule-based logic that isn't obviously rule-based

Two decisions look like they might benefit from a model but don't:

### Group column detection (in `eda.py`)

The heuristic: a column is a group candidate if it has between 10 and n_rows/3 unique values, with an average of 3+ rows per unique value. This catches user_id, session_id, patient_id patterns without false-positiving on IDs (fully unique) or on low-cardinality categoricals. An LLM prompted with column names could guess, but pattern-matching column names is fragile ("uid" vs "user_id" vs "usr" vs "userId"). The statistical test is more reliable.

### Text-column detection (in `eda.py`)

A column is text if its average string length exceeds 40 characters and its uniqueness ratio exceeds 0.5. This distinguishes free-text fields ("review", "description") from short categorical strings ("category", "state") without needing the LLM to look at samples.

## Testing philosophy

The rule-based layers get comprehensive unit tests. Every combination in the overfitting risk table is exercised. Every branch in the validation decision tree is exercised. Every dialogue transition is exercised. Total: 89 tests, running in about a second.

The LLM layer is tested only for orchestration (does `coach.run_session()` call the right module in the right order and hand it the right data?) — not for output content. Content is checked by the retrieval evaluator in `tests/run_eval.py`, which asserts that grounded questions retrieve their expected source files. That's the closest thing to a "does the LLM produce useful output?" check that doesn't require a live model.

## What's deliberately out of scope

- **Automated feature engineering.** KaggleCoach recommends techniques; it doesn't build features for you. Generated features are hard to test and easy to leak.
- **Model training loops.** The recommended baselines are code snippets in the knowledge base. Running them in-app would need dataset-specific plumbing that's out of scope for a strategic advisor.
- **Live leaderboard connectivity.** Competition-specific integrations (Kaggle API, DrivenData, etc.) belong in a separate tool. KaggleCoach is deliberately competition-platform-agnostic.

## Future extensions

The three-collection design was chosen partly to make growth cheap. Adding a fourth collection (say, `image`) is:

1. Create `knowledge/image/*.md` documents.
2. Add `image = "knowledge/image"` under `[collections]` in `config.toml`.
3. Re-run `python -m kagglecoach.indexer`.
4. Extend `primary_collection()` in `dialogue.py` to route image-task answers.

No changes to the retriever, coach, or renderer. That was the design goal, and it's the property that makes the current structure feel worth its complexity even at 12 documents.
