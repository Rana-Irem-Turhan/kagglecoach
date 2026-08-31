"""
Retrieval evaluator.

Runs the questions from `eval_questions.py` against a live
`MultiCollectionStore` (populated by the indexer) and reports how many
grounded questions hit the expected source files.

Usage:
    python tests/run_eval.py           # requires indexed store
    python tests/run_eval.py --embed-only   # embed and score against seeded data

The script does not require Foundry Local to be running for grounded
scoring — it uses the ModelClient normally, so make sure the store has
been indexed once beforehand (`python -m kagglecoach.indexer`).

Exit code is 0 when at least 70% of grounded questions retrieve their
expected source. Otherwise 1, for CI use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kagglecoach.dialogue import primary_collection
from kagglecoach.models import ModelClient
from kagglecoach.retriever import MultiCollectionRetriever
from kagglecoach.settings import SETTINGS
from kagglecoach.store import MultiCollectionStore
from tests.eval_questions import ALL_QUESTIONS, EvalQuestion


PASS_THRESHOLD = 0.70


def _score_grounded(question: EvalQuestion, hits) -> tuple[bool, str]:
    if not hits:
        return False, "no hits"
    if not question.expected_source_contains:
        # No source constraint — pass on any hit.
        return True, "any hit accepted"
    got_sources = {h.source_file.lower() for h in hits}
    for expected in question.expected_source_contains:
        if any(expected.lower() in src for src in got_sources):
            return True, f"matched {expected}"
    return False, f"no match; got {got_sources}"


def _score_out_of_scope(question: EvalQuestion, hits, result) -> tuple[bool, str]:
    # Success == retrieval confidence gate correctly trips
    if not result.confidence_ok:
        return True, "confidence gate correctly rejected"
    # Or at least, no strong hits at all
    if len(hits) == 0:
        return True, "no strong hits"
    return False, f"unexpected: {len(hits)} hits with confidence_ok=True"


def _score_edge(question: EvalQuestion, hits) -> tuple[bool, str]:
    if not question.expected_terms:
        # Just want the system to gracefully return no hits
        return len(hits) == 0, ("passed" if not hits else "hits when none expected")
    # Return true if at least one hit contains any expected term
    for h in hits:
        body = h.body.lower()
        if any(term.lower() in body for term in question.expected_terms):
            return True, "expected term found"
    return False, "no expected term in hits"


def run_eval(verbose: bool = True) -> tuple[int, int, int]:
    """Run the evaluation. Returns (grounded_pass, oos_pass, edge_pass)."""
    grounded_pass = 0
    oos_pass = 0
    edge_pass = 0

    grounded_total = sum(1 for q in ALL_QUESTIONS if q.category == "grounded")
    oos_total = sum(1 for q in ALL_QUESTIONS if q.category == "out_of_scope")
    edge_total = sum(1 for q in ALL_QUESTIONS if q.category == "edge")

    with MultiCollectionStore(SETTINGS.db_path, SETTINGS.faiss_dir) as store:
        if store.size() == 0:
            print("Store is empty. Run `python -m kagglecoach.indexer` first.")
            return 0, 0, 0

        client = ModelClient(SETTINGS)
        retriever = MultiCollectionRetriever(store=store, model_client=client)

        for q in ALL_QUESTIONS:
            # Pick a primary collection matching the question's flavour.
            if q.category == "grounded":
                if any(w in q.query.lower() for w in ("lightgbm", "catboost", "xgboost", "target encoding")):
                    primary = "tabular"
                elif any(w in q.query.lower() for w in ("tf-idf", "transformer", "imbalance nlp", "bert")):
                    primary = "nlp"
                else:
                    primary = "general_ml"
            else:
                primary = "general_ml"

            result = retriever.find_across(q.query, primary)
            all_hits = [h for hits in result.by_collection.values() for h in hits]

            if q.category == "grounded":
                ok, why = _score_grounded(q, all_hits)
                if ok:
                    grounded_pass += 1
            elif q.category == "out_of_scope":
                ok, why = _score_out_of_scope(q, all_hits, result)
                if ok:
                    oos_pass += 1
            else:
                ok, why = _score_edge(q, all_hits)
                if ok:
                    edge_pass += 1

            if verbose:
                mark = "✓" if ok else "✗"
                print(f"  {mark} [{q.category:12s}] {q.id:35s} {why}")

    print()
    print(f"Grounded:     {grounded_pass}/{grounded_total}")
    print(f"Out of scope: {oos_pass}/{oos_total}")
    print(f"Edge:         {edge_pass}/{edge_total}")

    return grounded_pass, oos_pass, edge_pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    gp, op, ep = run_eval(verbose=not args.quiet)
    grounded_total = sum(1 for q in ALL_QUESTIONS if q.category == "grounded")

    if grounded_total > 0 and gp / grounded_total >= PASS_THRESHOLD:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
