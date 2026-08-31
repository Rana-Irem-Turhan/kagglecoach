"""
Multi-collection retriever.

Wraps `MultiCollectionStore` with the retrieval-specific policies:
similarity threshold filtering, primary-collection confidence gating,
and cross-collection search that returns hits keyed by collection name.

Confidence gating:
    Chunks below `similarity_threshold` are dropped (not shown as weak
    matches). The primary collection also has a `min_strong_hits_primary`
    check — when the coach requests reasoning about the primary domain
    and fewer strong hits come back, the coach lowers confidence and the
    report notes reduced grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kagglecoach.models import ModelClient
from kagglecoach.settings import SETTINGS, Settings
from kagglecoach.store import Chunk, MultiCollectionStore


@dataclass(frozen=True)
class Match:
    """One retrieved chunk with its similarity score and collection name."""

    chunk: Chunk
    score: float

    @property
    def source_file(self) -> str:
        return self.chunk.source_file

    @property
    def body(self) -> str:
        return self.chunk.body

    @property
    def collection(self) -> str:
        return self.chunk.collection


@dataclass(frozen=True)
class RetrievalResult:
    """Grouped results — matches keyed by collection name."""

    by_collection: dict[str, list[Match]]
    primary_collection: str
    primary_strong_hits: int
    confidence_ok: bool
    dropped_below_threshold: int


class MultiCollectionRetriever:
    """
    Retrieves grounded evidence from the three FAISS collections.

    Two entry points:
        * `find(query, collection)` — search one collection.
        * `find_across(query, primary)` — search all collections and
          return matches grouped by collection, with confidence gating
          applied to the primary.
    """

    def __init__(
        self,
        store: MultiCollectionStore,
        model_client: ModelClient,
        settings: Settings = SETTINGS,
    ) -> None:
        self.store = store
        self.model_client = model_client
        self.settings = settings

    def find(
        self,
        query: str,
        collection: str,
        top_k: Optional[int] = None,
    ) -> list[Match]:
        """Search a single collection."""
        if not query.strip():
            return []
        top_k = top_k or self.settings.top_k_per_collection
        query_vec = self.model_client.embed_query(query)
        raw = self.store.search(collection, query_vec, top_k)
        return [Match(chunk=c, score=s) for c, s in raw]

    def find_across(
        self,
        query: str,
        primary_collection: str,
        top_k_per_collection: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Search all three collections and apply confidence gating.

        The primary collection is the one dictated by the user's task type
        (tabular / nlp). It gets the strong-hit check; the others contribute
        supporting evidence without gating.
        """
        top_k = top_k_per_collection or self.settings.top_k_per_collection
        threshold = self.settings.similarity_threshold

        if not query.strip():
            return RetrievalResult(
                by_collection={},
                primary_collection=primary_collection,
                primary_strong_hits=0,
                confidence_ok=False,
                dropped_below_threshold=0,
            )

        query_vec = self.model_client.embed_query(query)

        by_collection: dict[str, list[Match]] = {}
        dropped = 0

        for coll_name in self.settings.collections:
            raw = self.store.search(coll_name, query_vec, top_k)
            kept: list[Match] = []
            for chunk, score in raw:
                if score >= threshold:
                    kept.append(Match(chunk=chunk, score=score))
                else:
                    dropped += 1
            if kept:
                by_collection[coll_name] = kept

        primary_hits = by_collection.get(primary_collection, [])
        primary_strong_hits = sum(
            1 for m in primary_hits if m.score >= threshold
        )
        confidence_ok = primary_strong_hits >= self.settings.min_strong_hits_primary

        return RetrievalResult(
            by_collection=by_collection,
            primary_collection=primary_collection,
            primary_strong_hits=primary_strong_hits,
            confidence_ok=confidence_ok,
            dropped_below_threshold=dropped,
        )


def format_evidence_block(result: RetrievalResult, max_chars_per_chunk: int = 900) -> str:
    """
    Serialise retrieval results into the EVIDENCE block the LLM sees.

    Order: primary collection first, then supporting collections. Chunks
    are labelled with source filename so the model can cite them.
    """
    if not result.by_collection:
        return "(no evidence retrieved)"

    lines: list[str] = []
    ordered = [result.primary_collection] + [
        c for c in result.by_collection if c != result.primary_collection
    ]
    for coll_name in ordered:
        matches = result.by_collection.get(coll_name)
        if not matches:
            continue
        lines.append(f"### collection: {coll_name}")
        for i, m in enumerate(matches, start=1):
            body = m.body
            if len(body) > max_chars_per_chunk:
                body = body[:max_chars_per_chunk].rstrip() + " …"
            lines.append(
                f"[{coll_name}/{m.source_file} · score={m.score:.2f}]\n{body}"
            )
        lines.append("")
    return "\n".join(lines).strip()
