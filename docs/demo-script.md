# KaggleCoach demo script

A 5-minute walk-through for a live demo, screencast, or self-guided tour. Assumes Foundry Local is running and the indexer has populated the FAISS collections.

## Setup (do before recording)

```powershell
# From the project root
python -m kagglecoach.indexer     # ~60 seconds
streamlit run kagglecoach\ui.py
```

The browser opens `http://localhost:8501`.

## Scene 1 — What KaggleCoach is (30 seconds)

**On screen:** empty Streamlit page with the sidebar visible.

**Say:**

> KaggleCoach is an interactive strategy agent for tabular and NLP competitions. It runs Foundry Local for embeddings and chat by default — nothing about your dataset leaves the machine. The sidebar shows the three knowledge collections it retrieves from: tabular strategy, NLP strategy, and general ML foundations.

Point to each collection status pill in the sidebar. Highlight the 🟢 indicators.

## Scene 2 — The dialogue (90 seconds)

**On screen:** left panel showing question 1 of 7.

**Say:**

> Instead of asking one open-ended question, KaggleCoach walks through a short branching dialogue. Each answer routes to the next question — a state machine, not an LLM chain. That means the flow is deterministic and testable.

Click through the questions rapidly:

- **Q1:** Tabular classification
- **Q2:** ROC AUC
- **Q3:** Medium (10k – 100k rows)
- **Q4:** CPU only
- **Q5:** 1 – 4 weeks
- **Q6:** Some experience with similar problems
- **Q7:** Balanced — some learning, decent placement

Note that the metric options in Q2 changed based on the task type answered in Q1.

**Say:**

> Watch the metric options — I picked tabular classification, so I get log loss, ROC AUC, F1. If I'd picked regression, I'd see RMSE, MAE, RMSLE. The branching happens because different task families reward different strategies.

## Scene 3 — Upload a CSV (60 seconds)

**On screen:** sidebar file uploader.

Upload `examples/sample_titanic.csv` — 891 rows, mixed types.

**Say:**

> KaggleCoach can work purely from the dialogue, but if you upload a CSV, it also runs a dataset profile — column types, missingness, cardinality, class balance, group column detection. This is pure pandas, no model involved.

Pick `Survived` as the target column from the sidebar dropdown.

Add optional train/validation scores: **0.94** and **0.83**.

**Say:**

> I'll also enter train and validation scores from a hypothetical baseline — 0.94 on train, 0.83 on validation. A gap of 0.11 will flag as mild overfitting; combined with the small dataset, this pushes the risk to high.

## Scene 4 — The report (2 minutes)

**On screen:** right panel showing the strategy report as it materialises.

Scroll through each section:

### Section 1-2: Context and dataset profile

**Say:**

> The report starts with everything KaggleCoach knows about the problem — user answers on top, dataset profile below. Notice it detected `Sex`, `Pclass`, and `Embarked` as categorical, `Age` and `Fare` as numeric, and flagged `Age` for moderate missingness.

### Section 3: Recommended pipeline

**Say:**

> A five-phase pipeline scaffold from Understand to Ensemble. This is rule-based — the same five phases every time, but the model class in Phase 2 changes based on the task family, and the validation strategy comes from the decision tree.

### Sections 4-5: Model selection and feature engineering (LLM sections)

**Say:**

> These two sections are where the local model actually generates prose. The system prompt forbids inventing numbers — everything the model says should trace back to evidence retrieved from the knowledge base. Look at the citations — you can see it's pulling from `01-lightgbm-strategy.md` and `04-tabular-feature-engineering.md`.

### Section 6: Overfitting assessment

**Say:**

> Overfitting classification is fully rule-based. The gap of 0.11 puts us in the mild bucket by our thresholds, and combined with the small dataset it lifts the overall risk to high. The signals below explain the reasoning.

### Section 7: Validation strategy

**Say:**

> StratifiedKFold for classification with mild imbalance. The warning tells you not to use accuracy on this problem — a signal from the metric-implications knowledge document.

### Section 8: Retrieved evidence

**Say:**

> Every source that fed the LLM sections is listed here, with the best similarity score for each. If you're skeptical about a recommendation, this is where to trace it back.

## Scene 5 — The Azure OpenAI toggle (30 seconds)

**On screen:** sidebar toggle labelled "Use Azure OpenAI for chat".

Toggle it on.

**Say:**

> One more thing worth noting — this toggle switches chat generation to Azure OpenAI, without moving embeddings off the machine. Only the chat prompt goes to Azure; the dataset stays local. Useful for higher-quality prose when you're on a smaller local model, without giving up the privacy properties for the data itself.

Toggle it back off. Click "Start a new session" in the sidebar to reset.

## Scene 6 — Design closing (30 seconds)

**Say:**

> The design philosophy is that the LLM only does what it's good at — reading retrieved evidence and turning it into prose. Everything statistical is deterministic Python code. That's why the report has verifiable trust-anchors: numbers come from rule-based analysis you can inspect line by line, prose comes from the knowledge base you can read directly, and citations tie the two together.

## What to prepare beforehand

- `foundry model list` output visible somewhere on your desktop for the "Foundry Local is running" moment.
- `examples\sample_titanic.csv` in the outputs folder so upload is one click.
- Streamlit terminal visible in a corner so viewers can see the app is genuinely local.
- A clean browser window with no other tabs.

## Total runtime target

5 minutes 15 seconds including intro. Cut Scene 5 if you need to hit 5 minutes flat.
