# 🏆 KaggleCoach

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![Foundry Local](https://img.shields.io/badge/Microsoft-Foundry%20Local-0078D4?logo=microsoft)](https://learn.microsoft.com/azure/foundry-local/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)](https://github.com/facebookresearch/faiss)
[![sentence-transformers](https://img.shields.io/badge/sentence--transformers-Embeddings-green)](https://www.sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Interactive, dataset-aware ML competition strategy advisor powered by Microsoft Foundry Local and grounded retrieval.**

KaggleCoach is a strategic advisor for tabular and NLP machine-learning competitions. Users describe their problem through a short adaptive dialogue and can optionally upload a training CSV.

KaggleCoach then:

* profiles the dataset with deterministic pandas-based analysis,
* recommends a validation strategy,
* assesses overfitting risk,
* retrieves relevant evidence from specialised knowledge collections,
* and generates a structured Markdown strategy report.

The system uses a hybrid architecture: statistical decisions are handled by deterministic Python code, while the local language model turns retrieved evidence into readable strategy recommendations.

By default, chat generation runs locally through **Microsoft Foundry Local** using **Qwen2.5-1.5B Instruct**. RAG embeddings are generated locally with **sentence-transformers/all-MiniLM-L6-v2**.

```text
┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐
│ Adaptive        │  │ Dataset profile  │  │ Grounded report    │
│ branching       │→ │ overfitting risk │→ │ with source        │
│ dialogue        │  │ validation pick  │  │ citations          │
└─────────────────┘  └──────────────────┘  └────────────────────┘
    (rule-based)         (pure pandas)         (RAG + LLM prose)
```

---

## 📚 Knowledge Base

| Collection      | Files | Chunks | Topics                                                                                    |
| --------------- | ----: | -----: | ----------------------------------------------------------------------------------------- |
| 📊 `tabular`    |     5 |     34 | LightGBM, CatBoost, XGBoost, feature engineering, ensembles                               |
| 📝 `nlp`        |     3 |     23 | TF-IDF baselines, transformer fine-tuning, imbalance, pseudo-labeling                     |
| 🧪 `general_ml` |     4 |     41 | Validation strategies, metric implications, overfitting diagnosis, adversarial validation |

The knowledge base currently contains **12 documents and 98 indexed chunks**.

---

## ⚡ Quickstart — Windows

### 1. Prerequisites

Install Python and Microsoft Foundry Local.

```powershell
winget install Microsoft.FoundryLocal
winget install Python.Python.3.12
```

Python 3.11+ is recommended.

---

### 2. Install KaggleCoach

```powershell
cd kagglecoach

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

### 3. Start Foundry Local

KaggleCoach currently connects to Foundry Local through port `60458`.

```powershell
foundry server restart --port 60458 --idle-timeout 0
```

Check that the server is ready:

```powershell
foundry server status
```

Expected endpoint:

```text
http://127.0.0.1:60458
```

---

### 4. Download and load the local chat model

KaggleCoach uses the CUDA variant of **Qwen2.5-1.5B Instruct** for local chat generation.

```powershell
foundry model load qwen2.5-1.5b-instruct-cuda-gpu:4
```

Verify the loaded model:

```powershell
foundry model list --loaded -v
```

Expected model:

```text
Model Name: qwen2.5-1.5b
Device: GPU
Model ID: qwen2.5-1.5b-instruct-cuda-gpu:4
```

Qwen2.5-1.5B was selected as the default local model because it provides a better memory footprint for GPUs with limited VRAM while still supporting structured strategy generation.

---

### 5. Build the FAISS knowledge index

```powershell
python -m kagglecoach.indexer
```

The indexer generates embeddings locally using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and creates separate FAISS indexes for:

```text
tabular
nlp
general_ml
```

---

### 6. Launch the application

```powershell
streamlit run kagglecoach\ui.py
```

The browser normally opens at:

```text
http://localhost:8501
```

See [`docs/setup.md`](docs/setup.md) for additional setup and troubleshooting information.

> **Implementation note:** Chat generation uses Microsoft Foundry Local through its OpenAI-compatible HTTP API. RAG embeddings are generated independently with `sentence-transformers/all-MiniLM-L6-v2` and indexed using FAISS.

---

## 🔍 What Makes KaggleCoach Different

KaggleCoach is a **strategic advisor**, not a standard single-turn RAG chatbot.

| Dimension                | Typical RAG chatbot         | KaggleCoach                                          |
| ------------------------ | --------------------------- | ---------------------------------------------------- |
| **Interaction model**    | Single-turn question/answer | Multi-turn adaptive dialogue                         |
| **Vector storage**       | One collection              | Three specialised collections                        |
| **Backend**              | Single vector index         | FAISS `IndexFlatIP` per collection + SQLite metadata |
| **Reasoning**            | Mostly LLM inference        | Deterministic ML analysis + grounded LLM synthesis   |
| **Output**               | General text response       | Structured 8-section Markdown strategy report        |
| **Data-awareness**       | Text query only             | Optional CSV upload + pandas profiling               |
| **Dataset inspection**   | Usually none                | ✅                                                    |
| **Constraint awareness** | Limited                     | ✅ GPU, deadline, experience, goal                    |
| **Knowledge routing**    | Single corpus               | ✅ Task-aware collection routing                      |
| **Local inference**      | Optional                    | ✅ Microsoft Foundry Local                            |
| **Confidence control**   | Often implicit              | Explicit similarity threshold + strong-hit gate      |

The central design principle is simple:

> **Deterministic Python handles statistical decisions. The LLM handles grounded explanation and synthesis.**

This reduces the amount of numerical reasoning delegated to the language model and makes the generated strategy easier to trace back to code or retrieved evidence.

---

## ✨ Features

### 🧠 Hybrid Architecture

**Deterministic components**

* adaptive dialogue state machine,
* pandas dataset profiling,
* task-type detection,
* class-balance analysis,
* overfitting risk assessment,
* validation strategy selection,
* report scaffolding.

**LLM-assisted components**

* model-selection explanation,
* feature-engineering recommendations,
* evidence synthesis,
* readable strategy report generation.

---

### 📚 Three-Collection Retrieval

KaggleCoach separates its knowledge base into specialised collections:

```text
knowledge/tabular/
knowledge/nlp/
knowledge/general_ml/
```

Topics include:

**Tabular**

* LightGBM
* CatBoost
* XGBoost
* feature engineering
* ensembling

**NLP**

* TF-IDF baselines
* transformer fine-tuning
* text preprocessing
* imbalance handling
* pseudo-labeling

**General ML**

* cross-validation
* metric selection
* overfitting diagnosis
* adversarial validation

The first dialogue answer determines the **primary collection**.

Supporting collections may also provide cross-cutting evidence.

Retrieval uses:

* FAISS `IndexFlatIP`,
* normalised embeddings,
* similarity filtering,
* primary-collection confidence gating.

---

### 📊 Dataset-Aware Advice

Users can upload a CSV file to enable dataset-specific analysis.

KaggleCoach can inspect:

* number of rows and columns,
* numeric features,
* categorical features,
* text columns,
* missing values,
* cardinality,
* class distribution,
* likely task type,
* rows-per-feature ratio.

Optional train and validation scores can also be entered to estimate overfitting severity.

Current gap classifications are:

```text
healthy
mild
severe
```

The score gap is combined with dataset characteristics to produce an overall risk level.

---

### 🔒 Local-First Design

By default:

* chat generation runs through **Microsoft Foundry Local**,
* embeddings are generated locally with `sentence-transformers`,
* FAISS retrieval happens locally,
* SQLite metadata remains local,
* CSV analysis happens inside the local Python process.

The optional Azure OpenAI mode changes the chat backend only.

When Azure mode is enabled, the raw uploaded CSV remains local. However, dataset-profile information included in the generated prompt may be transmitted to the configured Azure OpenAI deployment.

---

## 🧩 Current Local AI Stack

```text
User
 │
 ▼
Streamlit UI
 │
 ▼
Adaptive dialogue
 │
 ├──────────────► pandas EDA
 │
 ├──────────────► validation rules
 │
 └──────────────► overfitting rules
 │
 ▼
Task-aware retrieval
 │
 ├── tabular FAISS index
 ├── nlp FAISS index
 └── general_ml FAISS index
 │
 ▼
sentence-transformers
all-MiniLM-L6-v2
 │
 ▼
Retrieved evidence
 │
 ▼
Microsoft Foundry Local
Qwen2.5-1.5B Instruct
 │
 ▼
Structured Markdown strategy report
```

---

## 📁 Repository Layout

```text
kagglecoach/
├── README.md
├── LICENSE
├── config.toml
├── requirements.txt
├── .env.example
├── .gitignore
│
├── knowledge/
│   ├── tabular/
│   ├── nlp/
│   └── general_ml/
│
├── kagglecoach/
│   ├── settings.py
│   ├── chunker.py
│   ├── store.py
│   ├── models.py
│   ├── indexer.py
│   ├── eda.py
│   ├── dialogue.py
│   ├── overfitting.py
│   ├── validation.py
│   ├── retriever.py
│   ├── coach.py
│   ├── report.py
│   └── ui.py
│
├── tests/
│   ├── test_chunker.py
│   ├── test_store.py
│   ├── test_eda.py
│   ├── test_dialogue.py
│   ├── test_overfitting.py
│   ├── test_validation.py
│   ├── test_retriever.py
│   ├── test_coach.py
│   ├── eval_questions.py
│   └── run_eval.py
│
├── docs/
│   ├── setup.md
│   ├── design.md
│   ├── architecture.md
│   ├── evaluation.md
│   └── demo-script.md
│
├── screenshots/
│
└── examples/
    └── sample_titanic.csv
```

---

## 🧪 Testing

### Unit tests

Unit tests do not require Foundry Local.

```powershell
python -m pytest tests -q
```

Current result:

```text
89 passed
```

The suite covers:

* chunking,
* vector storage,
* EDA,
* dialogue logic,
* overfitting assessment,
* validation selection,
* retrieval,
* orchestration.

---

### Retrieval Evaluation

Build the index first:

```powershell
python -m kagglecoach.indexer
```

Then run:

```powershell
python tests\run_eval.py
```

Current evaluation result:

```text
Grounded:     10/10
Out of scope: 3/3
Edge:         3/3
```

Total:

```text
16/16
```

The evaluation set contains:

* 10 grounded retrieval questions,
* 3 out-of-scope questions,
* 3 edge cases.

The evaluator checks whether grounded questions retrieve the expected knowledge source and whether unrelated questions are rejected by the confidence gate.

---

## ⚙️ Configuration

Runtime thresholds and retrieval settings are stored in [`config.toml`](config.toml).

Important sections include:

### `[retrieval]`

Controls:

* top-K retrieval,
* similarity threshold,
* primary-collection strong-hit requirements.

### `[collections]`

Maps knowledge folders to retrieval collections.

### `[eda]`

Controls:

* cardinality thresholds,
* missingness thresholds,
* dataset size buckets.

### `[overfitting]`

Controls:

* train-validation gap thresholds,
* rows-per-feature risk threshold.

The current local Foundry Local chat variant is selected in the model client.

---

## 📊 Example NLP Workflow

KaggleCoach can also analyse NLP datasets.

Example configuration:

```text
Task type: NLP classification
Metric: F1 (macro)
Data size: Small (< 10k rows)
Compute: Small GPU (< 8 GB VRAM)
Deadline: Less than a week
```

Example dataset profile:

```text
9,938 rows
2 columns
multiclass target
text feature detected
```

The resulting workflow can retrieve NLP evidence about:

* TF-IDF baselines,
* character and word n-grams,
* transformer fine-tuning,
* text preprocessing,
* validation,
* overfitting control.

---

## 🔧 Extending KaggleCoach

### Adding a knowledge collection

For example, to add time-series knowledge:

1. Create:

```text
knowledge/time_series/
```

2. Add Markdown knowledge files.

3. Add the collection to `config.toml`.

4. Extend task routing in `dialogue.py`.

5. Rebuild the index:

```powershell
python -m kagglecoach.indexer
```

The underlying FAISS store and report pipeline do not need to be rewritten.

---

### Adjusting thresholds

Risk and EDA thresholds can be modified in:

```text
config.toml
```

Relevant sections:

```text
[eda]
[overfitting]
[retrieval]
```

---

### Changing the Foundry Local model

Foundry Local provides multiple hardware-specific model variants.

List available variants with:

```powershell
foundry model list --variants -v
```

For machines with limited GPU VRAM, a smaller model can reduce out-of-memory risk.

The current tested GPU model is:

```text
qwen2.5-1.5b-instruct-cuda-gpu:4
```

On the development machine it used substantially less GPU memory than the previously tested Phi-3.5-mini CUDA model, leaving additional VRAM available for report generation.

---

## 🗺️ Roadmap

Potential future improvements:

* [ ] Time-series competition support
* [ ] Improved multiclass imbalance detection
* [ ] Task-aware retrieval weighting
* [ ] More NLP-specific model-selection logic
* [ ] Deep-learning strategy path
* [ ] Hyperparameter optimisation guidance
* [ ] Optuna integration
* [ ] Training-dynamics interpretation
* [ ] Loss-curve analysis
* [ ] General ML consulting mode
* [ ] Additional local model profiles

---

## 🎓 Microsoft AI Innovation Summer Internship

This project was developed as part of the **Microsoft AI Innovation Summer Internship Program**.

KaggleCoach demonstrates several applied AI engineering concepts:

* local language-model inference,
* retrieval-augmented generation,
* vector search,
* deterministic ML analysis,
* adaptive user interaction,
* structured report generation,
* optional cloud inference.

### Core technologies

* 🔵 **Microsoft Foundry Local** — local LLM inference
* 🔵 **Qwen2.5-1.5B Instruct** — current local chat model
* 🔵 **Azure OpenAI** — optional chat backend
* 🟠 **FAISS** — vector similarity search
* 🟢 **sentence-transformers** — local embeddings
* 🟣 **SQLite** — retrieval metadata
* 🔴 **Streamlit** — interactive UI
* 🐍 **pandas / Python** — deterministic dataset analysis

---

## 📄 Documentation

Additional documentation is available in:

* [`docs/setup.md`](docs/setup.md) — installation and troubleshooting
* [`docs/design.md`](docs/design.md) — architecture rationale
* [`docs/architecture.md`](docs/architecture.md) — system and data-flow diagrams
* [`docs/evaluation.md`](docs/evaluation.md) — testing and retrieval evaluation
* [`docs/demo-script.md`](docs/demo-script.md) — demonstration walkthrough

---

## 📄 License

MIT — see [`LICENSE`](LICENSE).

The knowledge base under `knowledge/` was authored specifically for KaggleCoach and contains general machine-learning strategy guidance.

---

## 🙏 Acknowledgements

Built with:

* **[Microsoft Foundry Local](https://learn.microsoft.com/azure/foundry-local/)** — local inference runtime
* **[FAISS](https://github.com/facebookresearch/faiss)** — vector similarity search
* **[sentence-transformers](https://www.sbert.net/)** — local embedding generation
* **[Streamlit](https://streamlit.io/)** — interactive application UI
* **[pandas](https://pandas.pydata.org/)** — deterministic dataset analysis
* **SQLite** — local metadata storage

---

<div align="center">

**Built using Microsoft Foundry Local · Qwen2.5 · FAISS · SQLite · Streamlit · Python**

[🐛 Report Bug](https://github.com/Rana-Irem-Turhan/kagglecoach/issues) · [💡 Request Feature](https://github.com/Rana-Irem-Turhan/kagglecoach/issues)

</div>
