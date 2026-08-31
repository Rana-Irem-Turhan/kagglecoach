"""
Rule-based dialogue engine.

The dialogue is a state machine, not an LLM chain. Each question is a
`Question` object; each answer routes to the next question through a
static branching table. This is deterministic, testable, and — most
importantly — reliable, unlike having a small local model decide the
next question from context.

Terminology:
    QuestionKey — enum-like string identifying which question is asked.
    Answer      — one of the option labels the user picked.
    Context     — accumulated {QuestionKey: Answer} dict built up over turns.

Public API:
    dialogue = Dialogue()
    q = dialogue.next_question(context)   # None when done
    dialogue.record_answer(context, key, answer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from kagglecoach.settings import SETTINGS, Settings


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Question:
    key: str
    prompt: str
    options: list[str]
    help_text: str = ""


DialogueContext = dict[str, str]


# --------------------------------------------------------------------------
# Question bank
# --------------------------------------------------------------------------
Q_TASK_TYPE = Question(
    key="task_type",
    prompt="What kind of competition or ML problem is this?",
    options=[
        "Tabular classification",
        "Tabular regression",
        "NLP classification",
        "NLP other (NER, generation, etc.)",
        "Other / not sure",
    ],
    help_text="The task family determines which knowledge base collection we'll retrieve from.",
)

Q_METRIC_CLASSIFICATION = Question(
    key="metric",
    prompt="What evaluation metric does the competition use?",
    options=["Log loss", "ROC AUC", "PR AUC", "F1", "Accuracy", "Other / not sure"],
    help_text="Different metrics reward different strategies — this changes the recommendations.",
)

Q_METRIC_REGRESSION = Question(
    key="metric",
    prompt="What evaluation metric does the competition use?",
    options=["RMSE", "MAE", "RMSLE", "MAPE", "R²", "Other / not sure"],
)

Q_METRIC_NLP = Question(
    key="metric",
    prompt="What evaluation metric does the competition use?",
    options=["Log loss", "ROC AUC", "F1 (macro)", "F1 (binary)", "Accuracy", "Other / not sure"],
)

Q_DATA_SIZE = Question(
    key="data_size",
    prompt="Roughly how large is the training dataset?",
    options=[
        "Small (< 10k rows)",
        "Medium (10k – 100k rows)",
        "Large (100k – 1M rows)",
        "Very large (> 1M rows)",
        "Not sure / no dataset yet",
    ],
)

Q_COMPUTE = Question(
    key="compute",
    prompt="What compute do you have available?",
    options=[
        "CPU only",
        "Small GPU (< 8 GB VRAM)",
        "Medium GPU (8 – 16 GB VRAM)",
        "Large GPU (> 16 GB VRAM)",
    ],
)

Q_DEADLINE = Question(
    key="deadline",
    prompt="How much time do you have until submission?",
    options=[
        "Less than a week",
        "1 – 4 weeks",
        "1 – 3 months",
        "No hard deadline",
    ],
)

Q_EXPERIENCE = Question(
    key="experience",
    prompt="How much experience do you have with this problem type?",
    options=[
        "First competition / new to this area",
        "Some experience with similar problems",
        "Very experienced",
    ],
)

Q_GOAL = Question(
    key="goal",
    prompt="What's your primary goal for this competition?",
    options=[
        "Leaderboard placement",
        "Learning a specific technique",
        "Balanced — some learning, decent placement",
    ],
)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
@dataclass
class Dialogue:
    """State machine that yields the next question based on prior answers."""

    settings: Settings = field(default_factory=lambda: SETTINGS)

    # -- entry point ------------------------------------------------------
    def next_question(self, context: DialogueContext) -> Optional[Question]:
        """Return the next question given the answers so far, or None if done."""
        if len(context) >= self.settings.max_questions:
            return None

        # The tree is a fixed sequence with branching at the metric step
        # based on task type.
        for step in self._build_steps(context):
            if step.key not in context:
                return step
        return None

    def record_answer(self, context: DialogueContext, key: str, answer: str) -> None:
        """Store an answer, validating that it belongs to the question's options."""
        current = self.next_question(context)
        if current is None:
            raise RuntimeError("Dialogue is complete; no question is awaiting an answer.")
        if current.key != key:
            raise ValueError(
                f"Answer to '{key}' does not match the awaited question '{current.key}'."
            )
        if answer not in current.options:
            raise ValueError(
                f"Answer '{answer}' is not among the valid options for '{key}': "
                f"{current.options}"
            )
        context[key] = answer

    def is_complete(self, context: DialogueContext) -> bool:
        return self.next_question(context) is None

    def progress(self, context: DialogueContext) -> tuple[int, int]:
        """Return (answered, total) — total counts the full branched tree."""
        total = len(self._build_steps(context))
        return len(context), total

    # -- private ----------------------------------------------------------
    def _build_steps(self, context: DialogueContext) -> list[Question]:
        """Assemble the ordered question sequence given current answers."""
        steps: list[Question] = [Q_TASK_TYPE]

        task = context.get("task_type")
        if task is None:
            # Only Q1 has been decided; return what we have — next call
            # after task_type is answered will extend the plan.
            return steps

        if task == "Tabular regression":
            steps.append(Q_METRIC_REGRESSION)
        elif task in ("NLP classification", "NLP other (NER, generation, etc.)"):
            steps.append(Q_METRIC_NLP)
        else:
            steps.append(Q_METRIC_CLASSIFICATION)

        steps.append(Q_DATA_SIZE)
        steps.append(Q_COMPUTE)
        steps.append(Q_DEADLINE)
        steps.append(Q_EXPERIENCE)
        steps.append(Q_GOAL)

        return steps


# --------------------------------------------------------------------------
# Routing helpers used elsewhere
# --------------------------------------------------------------------------
def primary_collection(context: DialogueContext) -> str:
    """Pick the primary RAG collection given the task type answer."""
    task = context.get("task_type", "")
    if task.startswith("Tabular"):
        return "tabular"
    if task.startswith("NLP"):
        return "nlp"
    return "general_ml"


def context_summary(context: DialogueContext) -> str:
    """Compact, human-readable one-liner for the report."""
    parts = []
    for k in ("task_type", "metric", "data_size", "compute", "deadline", "experience", "goal"):
        if k in context:
            parts.append(f"{k}={context[k]}")
    return "  ·  ".join(parts) if parts else "no dialogue answers"
