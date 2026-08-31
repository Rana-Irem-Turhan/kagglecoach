# KaggleCoach architecture

## System diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                                Streamlit UI                            │
│  ┌───────────────┐   ┌──────────────────┐   ┌────────────────────────┐ │
│  │ Left sidebar  │   │ Dialogue (main)  │   │ Strategy report        │ │
│  │ • mode toggle │   │ • one Q at a time│   │ • rule-based sections  │ │
│  │ • KB status   │   │ • progress bar   │   │ • LLM prose sections   │ │
│  │ • CSV upload  │   │ • radio + submit │   │ • evidence citations   │ │
│  │ • scores      │   └──────────────────┘   │ • download as markdown │ │
│  └───────────────┘                          └────────────────────────┘ │
└───────────────────┬────────────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                             Coach orchestrator                         │
│                                                                        │
│  1. eda.profile_dataframe(csv)      → DatasetProfile                   │
│  2. overfitting.assess_overfitting  → RiskAssessment                   │
│  3. validation.recommend_validation → ValidationRecommendation         │
│  4. retriever.find_across(query)    → RetrievalResult                  │
│  5. model_client.chat(system, user) → NL prose for two sections        │
│  6. Wrap all of the above into a StrategyReport                        │
└───────────────────┬─────────────────────────────┬──────────────────────┘
                    │                             │
                    ▼                             ▼
┌───────────────────────────────┐   ┌────────────────────────────────────┐
│  MultiCollectionRetriever     │   │  ModelClient                       │
│  • confidence threshold gate  │   │  • embed()  → Foundry Local        │
│  • primary strong-hit check   │   │  • chat()   → Foundry Local OR     │
│  • cross-collection merge     │   │              Azure OpenAI          │
└──────────┬────────────────────┘   └──────────────┬─────────────────────┘
           │                                       │
           ▼                                       │
┌───────────────────────────────┐                  │
│  MultiCollectionStore         │                  │
│  ┌─────────────────────────┐  │                  │
│  │ SQLite: chunks metadata │  │                  │
│  │  (id, collection,       │  │                  │
│  │   source_file, body,    │  │                  │
│  │   faiss_row, ...)       │  │                  │
│  └─────────────────────────┘  │                  │
│  ┌─────────────────────────┐  │                  │
│  │ FAISS IndexFlatIP × 3   │  │                  │
│  │  tabular.faiss          │  │                  │
│  │  nlp.faiss              │  │                  │
│  │  general_ml.faiss       │  │                  │
│  └─────────────────────────┘  │                  │
└───────────────────────────────┘                  │
           ▲                                       │
           │                                       ▼
┌──────────┴──────────────────────────────────────────────────────┐
│                       Indexer (offline)                         │
│                                                                 │
│  chunker.chunk_by_sections   ← knowledge/{tabular,nlp,general}/ │
│  model_client.embed(chunks)  → Foundry Local (embeddings only)  │
│  store.add_collection(...)   → writes FAISS + SQLite            │
└─────────────────────────────────────────────────────────────────┘
```

## Data flow: interactive session

1. **User loads the Streamlit app.**  UI initialises `st.session_state` and reads knowledge-base status from the store's `stats()` method.

2. **User answers each dialogue question.**  The state machine in `dialogue.py` validates the answer against the current question's options and stores it in `session_state.dialogue_context`. After the answer, `next_question(context)` recomputes the next step given the branching table.

3. **When all 7 questions are answered**, the UI calls `Coach.run_session()`:

   a. If a CSV was uploaded, `eda.profile_dataframe()` runs on it. Result: `DatasetProfile` with column types, missingness, cardinality, target task type, class balance, group candidates.

   b. `overfitting.assess_overfitting()` combines the profile's rows-per-feature ratio with (optional) train/val scores. Result: `RiskAssessment` with a level (low/medium/high) and a list of signals.

   c. `validation.recommend_validation()` walks the decision tree: datetime → TimeSeriesSplit; group columns → GroupKFold variant; classification imbalance → StratifiedKFold; else → KFold. Result: `ValidationRecommendation`.

   d. Two RAG queries are formulated: one for model selection, one for feature engineering. The `MultiCollectionRetriever` embeds each query and searches all three FAISS collections. Chunks below `similarity_threshold=0.30` are dropped. If the primary collection has fewer than `min_strong_hits_primary=2` strong hits, a low-confidence warning is added.

   e. The model client is called twice — once for the model-selection section, once for feature engineering. Each call includes the dataset profile summary + dialogue context + retrieved evidence in the prompt. The system prompt instructs the model to cite source filenames and refuse to invent numbers.

   f. Everything is packed into a `StrategyReport` dataclass and handed to `report.render_report()`, which produces the Markdown document.

4. **UI displays the Markdown** and offers a download button.

## Data flow: indexing

Indexing runs once (or after knowledge base edits) via `python -m kagglecoach.indexer`:

1. For each collection folder listed in `[collections]` in `config.toml`:
   1. Read every `.md` file in the folder.
   2. Chunk each file at H2/H3 headers, respecting `min_chars` and `max_chars` from `config.toml`.
   3. Batch-embed all chunks in one API call to Foundry Local.
   4. Wipe the collection's existing FAISS index and SQLite rows (unless `--keep`).
   5. Build a fresh `IndexFlatIP` from the L2-normalised embeddings.
   6. Write the index to `data/faiss/<collection>.faiss`.
   7. Insert one SQLite row per chunk with `(collection, source_file, section_idx, body, faiss_row)`.

Retrieval at query time is the inverse: FAISS returns row indices; SQLite resolves them to full chunk metadata.

## Module dependency graph

```
                  settings.py
                       │
        ┌──────────────┼──────────────┐
        │              │              │
     chunker.py    store.py       models.py
        │              │              │
        └──────┬───────┴──────┬───────┘
               │              │
           indexer.py      retriever.py ─────┐
                              │              │
                          coach.py           │
                        ┌─────┼─────┐        │
                        │     │     │        │
                     eda.py dialogue overfitting
                                  │        │
                              validation   │
                                     │     │
                                  report.py
                                     │
                                   ui.py
```

Bottom-to-top dependency: no module imports from a module below it in the graph.

## Configuration surface

Every runtime knob lives in `config.toml`. The `settings.py` module loads it once at import time into a frozen `Settings` dataclass. Every other module accepts an explicit `Settings` argument for testability, defaulting to the module-level singleton.

Tunable groups:

- `[models]` — Foundry Local model aliases; edit if your local catalogue uses different names.
- `[chunking]` — chunk size bounds.
- `[retrieval]` — top-K, similarity threshold, primary strong-hits gate.
- `[collections]` — folder-to-collection mapping (add or remove collections here).
- `[generation]` — temperature and token cap for LLM calls.
- `[dialogue]` — max question cap (safety brake on branching).
- `[eda]` — cardinality, missingness, and size-bucket thresholds.
- `[overfitting]` — gap-severity thresholds, rows-per-feature risk floor.
- `[paths]` — where SQLite and FAISS files live.
- `[web]` — Streamlit port (informational only; use `streamlit run --server.port` to actually change it).
