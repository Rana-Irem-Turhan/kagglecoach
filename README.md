# 🎯 KaggleCoach

**Interactive, dataset-aware ML competition strategy — runs locally on Foundry Local, with grounded advice from three specialised knowledge collections.**

KaggleCoach is a strategic advisor for tabular and NLP competitions. You describe your problem through a short adaptive dialogue and (optionally) upload the training CSV; KaggleCoach analyses the dataset, picks a validation strategy, assesses overfitting risk, and generates a structured Markdown report with model-selection and feature-engineering recommendations. Every recommendation cites the specific knowledge document it came from.

Built on **Microsoft Foundry Local** — embeddings and chat generation both run on your machine by default. An Azure OpenAI toggle is available for chat when you want higher-quality prose, but the uploaded dataset never leaves the machine.

```
┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐
│ Adaptive        │  │ Dataset profile  │  │ Grounded report    │
│ branching       │→ │ overfitting risk │→ │ with source        │
│ dialogue        │  │ validation pick  │  │ citations          │
└─────────────────┘  └──────────────────┘  └────────────────────┘
    (rule-based)         (pure pandas)         (RAG + LLM prose)
```

---

## Quickstart (Windows)

```powershell
# 1. Prerequisites (one-time)
winget install Microsoft.FoundryLocal
winget install Python.Python.3.12

# 2. Install
cd kagglecoach
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Build the FAISS index (30-90 seconds)
python -m kagglecoach.indexer

# 4. Launch
streamlit run kagglecoach\ui.py
```

The browser opens at `http://localhost:8501`. See [`docs/setup.md`](docs/setup.md) for troubleshooting and Linux/macOS instructions.

---

## What makes KaggleCoach different

KaggleCoach is a **strategic advisor**, not a Q&A bot. Its architecture is intentionally different from a typical single-collection RAG system:

| Dimension | Typical RAG chatbot | KaggleCoach |
|---|---|---|
| **Interaction model** | Single-turn ask/answer | Multi-turn stateful dialogue |
| **Vector storage** | One collection | Three collections (`tabular` / `nlp` / `general_ml`) with routing |
| **Backend** | numpy or single-index vector store | FAISS `IndexFlatIP` per collection + SQLite metadata |
| **Reasoning** | Pure LLM inference | Rule-based (EDA + validation + overfitting) + LLM prose |
| **Output** | Text answer | Structured 8-section Markdown report with citations |
| **Data-awareness** | Text queries only | Text + CSV upload → pandas profile → dataset-specific advice |
| **Confidence** | Implicit in response | Explicit primary-collection strong-hit gate |

The design philosophy is spelled out in [`docs/design.md`](docs/design.md): **the LLM only does what it's good at (turning retrieved evidence into prose), and everything statistical is deterministic Python code.** This produces reports where numbers, thresholds, and strategy names trace back to specific code paths or knowledge documents — nothing is invented at render time.

---

## Features

### 🧠 Hybrid architecture (rule-based + LLM)

- **Deterministic:** dialogue state machine, pandas EDA, overfitting risk assessment, validation strategy selection, report scaffolding.
- **LLM-generated:** model-selection reasoning and feature-engineering suggestions, grounded in retrieved evidence with source citations.
- Every rule-based decision is unit-tested (89 tests, ~1 second).

### 📚 Three-collection retrieval

- `knowledge/tabular/` — LightGBM, CatBoost, XGBoost, feature engineering, ensembles.
- `knowledge/nlp/` — TF-IDF baselines, transformer fine-tuning, imbalance and pseudo-labelling.
- `knowledge/general_ml/` — validation strategies, evaluation metrics, overfitting diagnosis, adversarial validation.
- The dialogue's first answer routes to a **primary collection**; supporting collections contribute cross-cutting evidence.
- A **similarity threshold** drops weak hits; a **strong-hits gate** on the primary collection triggers a low-confidence warning when evidence is thin.

### 📊 Dataset-aware advice

- Upload a CSV to enable a full pandas profile: column typing, missingness, cardinality, class balance, group column detection.
- Optional train/validation scores classify overfitting severity (`healthy` / `mild` / `severe`) and combine with dataset shape for an overall risk level.

### 🔒 Privacy by default

- Foundry Local runs **both embeddings and chat** on your machine.
- The Azure OpenAI toggle switches only **chat**; embeddings and dataset statistics stay local.
- The `.env.example` template documents the three env vars for Azure mode without asking for real credentials.

### 🖥️ Windows-first, cross-platform

- Setup uses `winget` and PowerShell throughout.
- FAISS-CPU installs cleanly via `pip` on Windows for Python 3.8-3.12.
- Linux and macOS work identically once Foundry Local is available.

---

## Repository layout

```
kagglecoach/
├── README.md                         ← you are here
├── LICENSE                           ← MIT
├── config.toml                       ← every runtime knob
├── requirements.txt
├── .env.example                      ← Azure OpenAI template
├── .gitignore
│
├── knowledge/                        ← the RAG corpus (12 files)
│   ├── tabular/     (5 docs)
│   ├── nlp/         (3 docs)
│   └── general_ml/  (4 docs)
│
├── kagglecoach/                      ← Python package
│   ├── settings.py       config loader
│   ├── chunker.py        section-based markdown chunker
│   ├── store.py          FAISS + SQLite multi-collection store
│   ├── models.py         Foundry Local + Azure OpenAI client
│   ├── indexer.py        build the three FAISS collections
│   ├── eda.py            pandas dataset profiler   ┐
│   ├── dialogue.py       rule-based state machine   │  rule-based
│   ├── overfitting.py    risk assessment            │  layers
│   ├── validation.py     strategy selector         ┘
│   ├── retriever.py      cross-collection RAG with confidence gating
│   ├── coach.py          orchestrator: run_session() → StrategyReport
│   ├── report.py         Markdown renderer
│   └── ui.py             Streamlit UI
│
├── tests/                            ← 89 tests, ~1 second
│   ├── test_chunker.py      (10)
│   ├── test_store.py        (10)
│   ├── test_eda.py          (21)
│   ├── test_dialogue.py     (13)
│   ├── test_overfitting.py  (12)
│   ├── test_validation.py   (12)
│   ├── test_retriever.py    (6)
│   ├── test_coach.py        (5)
│   ├── eval_questions.py    ← retrieval eval prompt bank
│   └── run_eval.py          ← retrieval evaluator (needs indexed store)
│
├── docs/
│   ├── setup.md            Windows-first setup guide
│   ├── design.md           hybrid architecture rationale
│   ├── architecture.md     system diagram and data flow
│   ├── evaluation.md       89-test breakdown, retrieval eval
│   └── demo-script.md      5-minute walkthrough
│
├── screenshots/            drop your captures here (Windows tips inside)
│
└── examples/
    └── sample_titanic.csv  100-row synthetic sample for demo
```

---

## Testing

```powershell
# Unit tests (no Foundry Local needed, no network)
python -m pytest tests -q
# 89 passed in 0.71s

# Retrieval evaluation (requires indexed store + running Foundry Local)
python -m kagglecoach.indexer
python tests\run_eval.py
```

The unit test suite covers every rule-based decision boundary. The retrieval evaluator asks 16 questions (10 grounded + 3 out-of-scope + 3 edge) against the live index and checks that grounded questions retrieve their expected source files.

See [`docs/evaluation.md`](docs/evaluation.md) for a detailed breakdown of what each test covers and what the retrieval eval does not measure.

---

## Configuration

Every runtime parameter lives in [`config.toml`](config.toml). Notable knobs:

- **`[models]`** — Foundry Local aliases (edit to match your local catalogue).
- **`[retrieval]`** — top-K per collection, similarity threshold, primary strong-hits gate.
- **`[collections]`** — folder-to-collection mapping. Add a new folder + one line here to add a new domain.
- **`[eda]`** — cardinality, missingness, and size-bucket thresholds.
- **`[overfitting]`** — gap-severity thresholds, rows-per-feature risk floor.

The config file is fully commented. Edit and restart the app; nothing else is needed.

---

## Documentation

- [`docs/setup.md`](docs/setup.md) — installation, PowerShell troubleshooting, Azure OpenAI env vars.
- [`docs/design.md`](docs/design.md) — why hybrid, why three collections, what the LLM is and isn't for.
- [`docs/architecture.md`](docs/architecture.md) — system diagram, data flow, module dependency graph, configuration surface.
- [`docs/evaluation.md`](docs/evaluation.md) — test coverage, retrieval eval methodology, manual smoke test.
- [`docs/demo-script.md`](docs/demo-script.md) — 5-minute walkthrough for demos and screencasts.

---

## Extending

**Adding a new knowledge collection** (e.g., `image` or `time_series`) is the fast path:

1. Create `knowledge/<name>/*.md`.
2. Add `<name> = "knowledge/<name>"` under `[collections]` in `config.toml`.
3. Extend `primary_collection()` in `kagglecoach/dialogue.py` to route the new task family.
4. Run `python -m kagglecoach.indexer`.

No changes to the retriever, coach, or renderer.

**Adjusting risk thresholds** — edit the `[overfitting]` and `[eda]` sections in `config.toml`. Rules in `kagglecoach/overfitting.py` and `kagglecoach/eda.py` read from these values, so no code edits are needed.

**Swapping the chat model** — edit `[models].chat` to any Foundry Local alias. For Azure OpenAI, set `AZURE_OPENAI_DEPLOYMENT` and toggle Azure mode in the sidebar.

---

## Licence

MIT — see [`LICENSE`](LICENSE).

The knowledge base under `knowledge/` was authored specifically for this project. It contains general ML strategy guidance based on well-established competition practices; no third-party copyrighted content is embedded.

---

## Acknowledgements

Built for the Microsoft Foundry Local summer school. Uses:

- **[Foundry Local](https://learn.microsoft.com/azure/foundry-local/)** — Microsoft's local-first inference runtime.
- **[FAISS](https://github.com/facebookresearch/faiss)** — Facebook AI Similarity Search.
- **[Streamlit](https://streamlit.io/)** — the UI framework.
- **[pandas](https://pandas.pydata.org/)** — for every EDA operation.
