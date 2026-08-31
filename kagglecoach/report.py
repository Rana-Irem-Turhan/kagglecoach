"""
Report renderer.

Takes a `StrategyReport` from `coach.run_session()` and produces a
structured Markdown document. The rendering itself is a template — every
number and section comes from the rule-based analyses or the LLM prose.
Nothing is invented at render time.

The output is written both to the Streamlit UI and to a downloadable
`.md` file the user can archive alongside their competition notes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from textwrap import indent
from typing import Optional

from kagglecoach.coach import StrategyReport
from kagglecoach.eda import DatasetProfile


def render_report(report: StrategyReport, title: str = "KaggleCoach strategy report") -> str:
    """Compose the full Markdown report."""
    lines: list[str] = []

    # Header ------------------------------------------------------------
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
                 f"chat backend: **{report.generation_mode}** · "
                 f"primary collection: **{report.primary_collection}***")
    lines.append("")

    if report.warnings:
        lines.append("> **Notes on this run**")
        for w in report.warnings:
            lines.append(f"> - {w}")
        lines.append("")

    # 1) User context ---------------------------------------------------
    lines.append("## 1. User context")
    lines.append("")
    lines.append(_format_dialogue_summary(report.dialogue_summary))
    lines.append("")

    # 2) Dataset profile ------------------------------------------------
    lines.append("## 2. Dataset profile")
    lines.append("")
    if report.dataset_profile is None:
        lines.append("*No dataset uploaded — recommendations use dialogue answers only.*")
    else:
        lines.extend(_format_profile(report.dataset_profile))
    lines.append("")

    # 3) Recommended pipeline (rule-based) ------------------------------
    lines.append("## 3. Recommended pipeline")
    lines.append("")
    lines.extend(_format_pipeline_phases(report))
    lines.append("")

    # 4) Model selection (LLM) -----------------------------------------
    lines.append("## 4. Model selection")
    lines.append("")
    lines.append(report.model_selection_prose)
    lines.append("")

    # 5) Feature engineering (LLM) --------------------------------------
    lines.append("## 5. Feature engineering")
    lines.append("")
    lines.append(report.feature_engineering_prose)
    lines.append("")

    # 6) Overfitting assessment (rule-based) ----------------------------
    lines.append("## 6. Overfitting assessment")
    lines.append("")
    lines.append(f"**Risk level:** `{report.overfitting.level}`  ")
    if report.overfitting.gap_classification is not None:
        lines.append(f"**Gap classification:** `{report.overfitting.gap_classification}`  ")
    if report.overfitting.numeric_gap is not None:
        lines.append(f"**Train/val gap:** `{report.overfitting.numeric_gap:+.4f}`  ")
    if report.overfitting.rows_per_feature is not None:
        lines.append(f"**Rows per feature:** `{report.overfitting.rows_per_feature:.0f}`  ")
    lines.append("")
    lines.append("Signals:")
    lines.append("")
    for s in report.overfitting.signals:
        lines.append(f"- {s}")
    lines.append("")

    # 7) Validation strategy (rule-based) -------------------------------
    lines.append("## 7. Validation strategy")
    lines.append("")
    lines.append(f"**Recommended:** `{report.validation.strategy}` with "
                 f"`n_splits={report.validation.n_splits}`")
    lines.append("")
    lines.append(f"**Why:** {report.validation.reason}")
    lines.append("")
    if report.validation.warnings:
        lines.append("**Watch out for:**")
        lines.append("")
        for w in report.validation.warnings:
            lines.append(f"- {w}")
        lines.append("")

    # 8) Retrieved evidence --------------------------------------------
    lines.append("## 8. Retrieved evidence")
    lines.append("")
    lines.extend(_format_evidence(report))
    lines.append("")

    # Footer ------------------------------------------------------------
    lines.append("---")
    lines.append("")
    lines.append(f"*Retrieval pulled from `{report.primary_collection}` (primary) plus "
                 f"{len([c for c in report.retrieval.by_collection if c != report.primary_collection])}"
                 f" supporting collection(s). "
                 f"{report.retrieval.dropped_below_threshold} weak matches dropped "
                 f"below similarity threshold.*")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Section formatters
# --------------------------------------------------------------------------
def _format_dialogue_summary(summary: str) -> str:
    """Turn the coach's dialogue summary into a bullet list."""
    if not summary or summary == "no dialogue answers":
        return "*No dialogue answers on record.*"
    lines = []
    for pair in summary.split("  ·  "):
        if "=" in pair:
            k, v = pair.split("=", 1)
            lines.append(f"- **{k}:** {v}")
    return "\n".join(lines)


def _format_profile(p: DatasetProfile) -> list[str]:
    lines: list[str] = []
    lines.append(f"- **Shape:** {p.n_rows:,} rows × {p.n_columns} columns ({p.size_bucket})")
    lines.append(f"- **Memory:** {p.memory_usage_mb:.1f} MB")

    if p.target_column:
        lines.append(f"- **Target:** `{p.target_column}` ({p.task_type})")
        if p.class_balance:
            balance = ", ".join(f"{k}={v:.1%}" for k, v in sorted(
                p.class_balance.items(), key=lambda kv: -kv[1]
            )[:5])
            lines.append(f"  - Class balance: {balance}")
            lines.append(f"  - Imbalance severity: **{p.imbalance_severity}**")
    else:
        lines.append("- **Target:** not specified")

    lines.append(f"- **Numeric columns ({len(p.numeric_columns)}):** "
                 + (", ".join(f"`{c}`" for c in p.numeric_columns[:8])
                    + (f" and {len(p.numeric_columns) - 8} more"
                       if len(p.numeric_columns) > 8 else "")))

    if p.categorical_columns:
        lines.append(f"- **Categorical columns ({len(p.categorical_columns)}):** "
                     + ", ".join(f"`{c}`" for c in p.categorical_columns[:8]))
    if p.high_cardinality_categorical:
        lines.append(f"  - High-cardinality (target encoding candidates): "
                     + ", ".join(f"`{c}`" for c in p.high_cardinality_categorical))
    if p.datetime_columns:
        lines.append(f"- **Datetime columns:** "
                     + ", ".join(f"`{c}`" for c in p.datetime_columns))
    if p.text_columns:
        lines.append(f"- **Text columns:** "
                     + ", ".join(f"`{c}`" for c in p.text_columns))
    if p.id_like_columns:
        lines.append(f"- **ID-like columns (usually drop before modelling):** "
                     + ", ".join(f"`{c}`" for c in p.id_like_columns))

    if p.columns_with_severe_missing:
        lines.append(f"- **Severe missingness:** "
                     + ", ".join(f"`{c}` ({p.missing_rates[c]:.0%})"
                                 for c in p.columns_with_severe_missing[:5]))
    elif p.columns_with_moderate_missing:
        lines.append(f"- **Moderate missingness:** "
                     + ", ".join(f"`{c}` ({p.missing_rates[c]:.0%})"
                                 for c in p.columns_with_moderate_missing[:5]))

    if p.likely_group_columns:
        lines.append(f"- **Likely group columns (leakage risk):** "
                     + ", ".join(f"`{c}`" for c in p.likely_group_columns))

    return lines


def _format_pipeline_phases(report: StrategyReport) -> list[str]:
    """Rule-based, five-phase pipeline scaffold."""
    p = report.dataset_profile
    primary = report.primary_collection
    val_name = report.validation.strategy

    phases = [
        (
            "Understand",
            [
                "Read the competition description end-to-end; note the metric, "
                "submission format, and any special rules.",
                "If you haven't already, generate a dataset profile "
                "(this report includes one above).",
            ],
        ),
        (
            "Baseline",
            [
                f"Set up {val_name} with `n_splits={report.validation.n_splits}` "
                "before writing any modelling code.",
                (
                    "Fit a minimal LightGBM baseline with default hyperparameters "
                    "and the raw features."
                    if primary == "tabular" else
                    "Fit a TF-IDF + logistic regression baseline with default hyperparameters."
                    if primary == "nlp" else
                    "Fit the simplest model appropriate to the task; measure the CV score."
                ),
                "Record the baseline score — every later change is judged against it.",
            ],
        ),
        (
            "Feature engineering",
            [
                "Apply Tier-1 features first (encoding by cardinality, datetime "
                "decomposition, missing indicators).",
                "Refit the baseline model after each batch of features to catch "
                "regressions early.",
            ],
        ),
        (
            "Tune and diversify",
            [
                (
                    "Tune LightGBM in priority order: num_leaves → min_data_in_leaf → "
                    "learning_rate. Use early stopping."
                    if primary == "tabular" else
                    "Tune TF-IDF `ngram_range` and `min_df`; sweep classifier `C` on log scale."
                    if primary == "nlp" else
                    "Tune the hyperparameters most relevant to the chosen model class."
                ),
                "Add a second model family for algorithmic diversity.",
            ],
        ),
        (
            "Ensemble and finalise",
            [
                "Average predictions from your top 2–3 diverse models; tune blend weights "
                "on CV, not on the public leaderboard.",
                "Verify the submission format against the sample submission file. "
                "Keep two independent submissions in case of last-minute issues.",
            ],
        ),
    ]

    out: list[str] = []
    for i, (name, bullets) in enumerate(phases, start=1):
        out.append(f"**Phase {i}: {name}**")
        out.append("")
        for b in bullets:
            out.append(f"- {b}")
        out.append("")
    return out


def _format_evidence(report: StrategyReport) -> list[str]:
    lines: list[str] = []
    if not report.retrieval.by_collection:
        lines.append("*No knowledge-base evidence was retrieved for this session.*")
        return lines

    for coll_name, files in report.evidence_summary.items():
        lines.append(f"**{coll_name}** — sources:")
        lines.append("")
        for filename in files:
            top_score: Optional[float] = None
            for m in report.retrieval.by_collection[coll_name]:
                if m.source_file == filename:
                    top_score = m.score if top_score is None else max(top_score, m.score)
            score_note = f" (best score: {top_score:.2f})" if top_score is not None else ""
            lines.append(f"- `{filename}`{score_note}")
        lines.append("")
    return lines
