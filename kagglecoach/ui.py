"""
Streamlit UI for KaggleCoach.

Run with:
    streamlit run kagglecoach/ui.py

Layout:
    * Left sidebar    : inference-mode toggle, KB status, CSV upload,
                        target-column picker, optional train/val scores.
    * Main panel      : dialogue (one question at a time with a progress
                        bar) followed by the generated report.

State lives in `st.session_state`:
    * dialogue_context   dict[str, str] of accumulated answers
    * uploaded_df        pandas DataFrame or None
    * report             StrategyReport once generated
    * report_markdown    rendered Markdown string for download

The UI writes no files. All persistence happens through the SQLite/FAISS
store the coach owns.
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

# Ensure the package is importable when running `streamlit run kagglecoach/ui.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from kagglecoach import __version__
from kagglecoach.coach import Coach
from kagglecoach.dialogue import Dialogue, context_summary
from kagglecoach.models import ModelClient
from kagglecoach.report import render_report
from kagglecoach.settings import SETTINGS
from kagglecoach.store import MultiCollectionStore


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _model_client() -> ModelClient:
    return ModelClient(SETTINGS)


@st.cache_resource(show_spinner=False)
def _store() -> MultiCollectionStore:
    return MultiCollectionStore(SETTINGS.db_path, SETTINGS.faiss_dir)


@st.cache_resource(show_spinner=False)
def _dialogue() -> Dialogue:
    return Dialogue(SETTINGS)


def _coach() -> Coach:
    """The Coach is cheap to build; recreate on each use so azure toggle takes effect."""
    return Coach(store=_store(), model_client=_model_client(), settings=SETTINGS)


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("dialogue_context", {})
    ss.setdefault("uploaded_df", None)
    ss.setdefault("uploaded_filename", None)
    ss.setdefault("target_column", None)
    ss.setdefault("train_score", None)
    ss.setdefault("val_score", None)
    ss.setdefault("higher_is_better", True)
    ss.setdefault("report_markdown", None)
    ss.setdefault("last_error", None)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### ⚙️ Inference mode")
        current_mode = _model_client().active_mode
        use_azure = st.toggle(
            "Use Azure OpenAI for chat",
            value=(current_mode == "azure"),
            help=("Off: Foundry Local runs chat generation on your machine. "
                  "On: chat goes to Azure OpenAI. Embeddings always stay local so "
                  "your uploaded dataset never leaves the machine."),
        )
        _model_client().use_azure_for_chat = use_azure
        st.caption(f"Active: **{_model_client().active_mode}**")

        st.divider()
        st.markdown("### 📚 Knowledge base status")
        store = _store()
        for coll in SETTINGS.collections:
            stats = store.stats(coll)
            emoji = "🟢" if stats.chunk_count > 0 else "🔴"
            st.caption(
                f"{emoji} `{coll}`: {stats.chunk_count} chunk(s) "
                f"from {stats.file_count} file(s)"
            )
        if store.size() == 0:
            st.warning(
                "The FAISS collections are empty. Run "
                "`python -m kagglecoach.indexer` from a terminal, then reload."
            )

        st.divider()
        st.markdown("### 📄 Dataset (optional)")
        uploaded = st.file_uploader(
            "Upload a CSV to enable dataset-aware advice",
            type=["csv"],
            help="The file stays on your machine — only summary statistics are used.",
        )
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.session_state.uploaded_df = df
                st.session_state.uploaded_filename = uploaded.name
                st.caption(f"Loaded `{uploaded.name}` — {len(df):,} rows × {df.shape[1]} cols")

                target = st.selectbox(
                    "Target column (optional)",
                    options=["(none)"] + list(df.columns),
                    index=0,
                    help="The column KaggleCoach should treat as the prediction target.",
                )
                st.session_state.target_column = None if target == "(none)" else target
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read the CSV: {exc}")

        st.divider()
        st.markdown("### 📊 Optional scores")
        col1, col2 = st.columns(2)
        with col1:
            train_input = st.text_input("Train score", value="", placeholder="e.g. 0.921")
        with col2:
            val_input = st.text_input("Val score", value="", placeholder="e.g. 0.874")
        st.session_state.train_score = _parse_float(train_input)
        st.session_state.val_score = _parse_float(val_input)
        st.session_state.higher_is_better = st.checkbox(
            "Higher score is better", value=True,
            help="Uncheck for loss-style metrics (log loss, RMSE).",
        )

        st.divider()
        st.markdown("### 🔄 Reset")
        if st.button("Start a new session", use_container_width=True):
            for k in ("dialogue_context", "uploaded_df", "uploaded_filename",
                      "target_column", "train_score", "val_score",
                      "report_markdown", "last_error"):
                st.session_state.pop(k, None)
            st.rerun()

        st.caption(f"KaggleCoach v{__version__}")


def _render_dialogue() -> None:
    dialogue = _dialogue()
    ctx = st.session_state.dialogue_context

    answered, total = dialogue.progress(ctx)
    st.progress(answered / max(total, 1),
                text=f"Question {min(answered + 1, total)} of {total}")

    q = dialogue.next_question(ctx)
    if q is None:
        st.success("All questions answered. Scroll down for the strategy report.")
        return

    st.markdown(f"#### {q.prompt}")
    if q.help_text:
        st.caption(q.help_text)

    # Radio buttons for the current question. Streamlit will trigger a rerun
    # when the value changes; we use a Submit button so accidental clicks
    # don't advance the state machine.
    choice = st.radio(
        label="",
        options=q.options,
        key=f"choice_{q.key}",
        label_visibility="collapsed",
    )
    submit = st.button("Submit answer", type="primary", use_container_width=True)
    if submit and choice:
        dialogue.record_answer(ctx, q.key, choice)
        st.rerun()


def _render_report_section() -> None:
    ctx = st.session_state.dialogue_context
    if not _dialogue().is_complete(ctx):
        st.info(
            "Answer the questions on the left to generate a strategy report. "
            "You can also upload a CSV in the sidebar for dataset-aware advice."
        )
        return

    if st.session_state.report_markdown is None:
        with st.status("Building your strategy report…", expanded=True) as status:
            try:
                st.write("→ Running dataset profile…")
                st.write("→ Assessing overfitting risk…")
                st.write("→ Selecting validation strategy…")
                st.write("→ Retrieving grounded evidence from FAISS…")
                st.write("→ Generating natural-language sections…")
                coach = _coach()
                report = coach.run_session(
                    dialogue_context=ctx,
                    uploaded_df=st.session_state.uploaded_df,
                    target_column=st.session_state.target_column,
                    train_score=st.session_state.train_score,
                    val_score=st.session_state.val_score,
                    higher_is_better=st.session_state.higher_is_better,
                )
                markdown = render_report(report)
                st.session_state.report_markdown = markdown
                status.update(label="Report ready.", state="complete", expanded=False)
            except Exception as exc:  # noqa: BLE001
                st.session_state.last_error = "".join(traceback.format_exception(exc))
                status.update(label=f"Report failed: {exc}", state="error")
                return

    st.markdown(st.session_state.report_markdown)
    filename = "kagglecoach-report.md"
    st.download_button(
        "📥 Download report as Markdown",
        data=st.session_state.report_markdown.encode("utf-8"),
        file_name=filename,
        mime="text/markdown",
        use_container_width=True,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _parse_float(raw: str) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="KaggleCoach",
        page_icon="🎯",
        layout="wide",
    )
    _init_state()

    st.title("🎯 KaggleCoach")
    st.caption(
        "Interactive, dataset-aware ML competition strategy — runs locally on "
        "Foundry Local, with grounded advice from three specialised knowledge collections."
    )

    _render_sidebar()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("### Dialogue")
        _render_dialogue()
        if st.session_state.dialogue_context:
            with st.expander("Your answers so far", expanded=False):
                st.text(context_summary(st.session_state.dialogue_context))
    with right:
        st.markdown("### Strategy report")
        _render_report_section()

    if st.session_state.last_error:
        with st.expander("Error details", expanded=False):
            st.code(st.session_state.last_error, language="text")


if __name__ == "__main__":
    main()
