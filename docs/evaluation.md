# KaggleCoach evaluation

Testing an LLM-adjacent tool requires two layers of evaluation: **deterministic tests** for the rule-based components, and **retrieval quality checks** for the RAG layer. Content quality of the LLM's output isn't tested by KaggleCoach directly — that's the responsibility of a downstream reviewer.

## Layer 1: unit tests (89 tests, ~1 second)

`pytest tests/` runs the full suite. Coverage by file:

| Test file | Count | What it covers |
|---|---|---|
| `test_chunker.py` | 10 | Header splitting, oversized fallback, merge rules, real KB files. |
| `test_store.py` | 10 | FAISS add/search, multi-collection independence, persistence. |
| `test_eda.py` | 21 | Column typing, size buckets, target detection, imbalance severity, group detection. |
| `test_dialogue.py` | 13 | State machine walks, answer validation, branching by task type. |
| `test_overfitting.py` | 12 | Every combination of dataset-shape signal + gap classification. |
| `test_validation.py` | 12 | Decision tree branches, dialogue-only fallback. |
| `test_retriever.py` | 6 | Multi-collection search, confidence gating, evidence formatting. |
| `test_coach.py` | 5 | End-to-end orchestration with the fake model client. |
| **Total** | **89** | |

Running the suite:

```powershell
python -m pytest tests -q
# 89 passed in 0.71s
```

### Fake model client

`tests/conftest.py::FakeModelClient` replaces `ModelClient` for orchestrator tests. Its embeddings are deterministic character histograms; its chat responses echo back the prompt length. This lets `test_coach.py` verify that:

- The right modules run in the right order.
- Retrieval is called with the right queries.
- The `StrategyReport` object is assembled correctly.
- The report renderer produces well-structured Markdown.

Without needing Foundry Local, without stochastic assertions, in under a second.

## Layer 2: retrieval evaluation (`tests/run_eval.py`)

The unit tests can't verify that the RAG layer surfaces the right documents for real questions. That's what `run_eval.py` measures.

### The question bank

`tests/eval_questions.py` defines three categories:

- **Grounded (10 questions)** — each has a specific expected source file. Example: "LightGBM baseline hyperparameters" should retrieve `01-lightgbm-strategy.md` in the top hits.
- **Out-of-scope (3 questions)** — deliberately outside KaggleCoach's domain (chocolate mousse, Roman emperors). Success means the confidence gate correctly declines to return strong hits.
- **Edge cases (3 questions)** — vague or ambiguous queries ("which model should I use", "overfitting"). Success means some sensible hit is retrieved.

### Running the eval

Requires a live indexed store, so run the indexer first:

```powershell
python -m kagglecoach.indexer
python tests\run_eval.py
```

Output:

```
✓ [grounded    ] grounded-lgbm-baseline              matched lightgbm
✓ [grounded    ] grounded-catboost-cats              matched catboost
✓ [grounded    ] grounded-target-encoding            matched feature-engineering
✓ [grounded    ] grounded-timeseries-validation      matched validation
...
✓ [out_of_scope] oos-recipe                          confidence gate correctly rejected
...

Grounded:     10/10
Out of scope: 3/3
Edge:         3/3
```

The runner exits with code 0 when ≥70% of grounded questions retrieve their expected source. Suitable for CI.

### What the eval does NOT measure

- **LLM output quality.** The retrieval eval checks that the right evidence is fetched, not that the model uses it well. A separate human review is the honest way to check this.
- **Report readability.** Grammar, tone, and formatting quality of LLM-generated prose sections need a reader, not a checker.
- **Numerical accuracy of recommendations.** Because the rule-based layers are the ones producing numbers, the unit tests already cover these. But the LLM can still misquote or paraphrase a hyperparameter incorrectly — a spot check on a fresh session is the fallback.

## Manual smoke test

The most reliable end-to-end check remains running through a real session:

1. Launch `streamlit run kagglecoach\ui.py`.
2. Answer the seven dialogue questions in a plausible pattern (e.g., Tabular classification / ROC AUC / Medium / CPU only / 1-4 weeks / Some experience / Balanced).
3. Upload `examples\sample_titanic.csv`.
4. Set target column to `Survived`.
5. Optionally add train/val scores (0.92 and 0.85 to trigger the mild-gap path).
6. Wait for the report.
7. Verify:
   - Dataset profile shows the right columns as categorical/numeric.
   - Validation strategy is StratifiedKFold with a warning about accuracy.
   - Overfitting level is medium (small dataset + mild gap).
   - Model selection cites `01-lightgbm-strategy.md` or `05-tabular-ensembles.md`.
   - No obviously fabricated hyperparameter values.

If any of those fail, the failure is in the specific corresponding layer and is easier to debug than "the LLM produced something weird".

## Regression guardrails

Anything that lands in `main` should:

- Pass `pytest tests -q` cleanly.
- Not regress `run_eval.py` from 10/10 grounded.
- Not change the schema in `store.py` without a migration path documented in the PR.

The `config.toml` thresholds are considered part of the public API. Changing them alters user-facing behaviour and needs a rationale in the changelog.
