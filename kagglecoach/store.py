"""
FAISS-backed multi-collection store.

Three separate FAISS indices (tabular, nlp, general_ml) live under
`faiss_dir/`. SQLite holds the chunk metadata (source file, section
index, body text) alongside a `chunk_id -> collection` mapping that lets
us resolve a FAISS hit back to its full record.

Why FAISS at this scale?
    The three collections combined are still under a thousand chunks —
    a numpy matmul would work fine. FAISS is used here because the spec
    calls for it, and because it makes the "add a new collection" path
    obvious: initialise another IndexFlatIP, dump it as a file, register
    it. When the corpus grows, the same code seamlessly moves from Flat
    to IVF or HNSW indices without changing the surrounding logic.

Vector normalisation:
    We store L2-normalised float32 vectors and use IndexFlatIP (inner
    product). On normalised vectors, inner product equals cosine
    similarity — same math as RubberDuck's manual matmul, just wrapped
    in FAISS's search API.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    collection    TEXT    NOT NULL,
    source_file   TEXT    NOT NULL,
    section_idx   INTEGER NOT NULL,
    body          TEXT    NOT NULL,
    body_length   INTEGER NOT NULL,
    embedding_dim INTEGER NOT NULL,
    faiss_row     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file);
"""


@dataclass(frozen=True)
class Chunk:
    """A single stored chunk with metadata (no embedding — that lives in FAISS)."""

    id: int
    collection: str
    source_file: str
    section_idx: int
    body: str
    body_length: int
    faiss_row: int


@dataclass(frozen=True)
class CollectionStats:
    name: str
    chunk_count: int
    file_count: int
    total_chars: int
    avg_chunk_chars: float
    embedding_dim: int | None


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise a batch of row vectors. Zero rows stay zero."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class MultiCollectionStore:
    """
    Owns the SQLite metadata table plus one FAISS index per collection.

    On construction it opens SQLite and lazily loads FAISS indices from
    disk on first access. Use as a context manager for clean shutdown.
    """

    def __init__(self, db_path: Path | str, faiss_dir: Path | str) -> None:
        self.db_path = Path(db_path)
        self.faiss_dir = Path(faiss_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

        # collection name -> FAISS index (loaded on demand)
        self._indices: dict[str, "faiss.Index"] = {}  # noqa: F821

    # -- lifecycle --------------------------------------------------------
    def __enter__(self) -> "MultiCollectionStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- FAISS helpers ----------------------------------------------------
    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "KaggleCoach needs FAISS. Install with: pip install faiss-cpu"
            ) from exc
        return faiss

    def _index_path(self, collection: str) -> Path:
        return self.faiss_dir / f"{collection}.faiss"

    def _load_index(self, collection: str) -> "faiss.Index | None":  # noqa: F821
        if collection in self._indices:
            return self._indices[collection]
        path = self._index_path(collection)
        if not path.exists():
            return None
        faiss = self._faiss()
        idx = faiss.read_index(str(path))
        self._indices[collection] = idx
        return idx

    def _save_index(self, collection: str) -> None:
        faiss = self._faiss()
        idx = self._indices.get(collection)
        if idx is None:
            return
        faiss.write_index(idx, str(self._index_path(collection)))

    # -- writes -----------------------------------------------------------
    def wipe(self) -> None:
        """Drop all chunks and delete FAISS indices from disk."""
        self._conn.execute("DELETE FROM chunks;")
        self._conn.commit()
        self._indices.clear()
        for path in self.faiss_dir.glob("*.faiss"):
            path.unlink()

    def wipe_collection(self, collection: str) -> None:
        """Drop only one collection."""
        self._conn.execute("DELETE FROM chunks WHERE collection = ?;", (collection,))
        self._conn.commit()
        self._indices.pop(collection, None)
        p = self._index_path(collection)
        if p.exists():
            p.unlink()

    def add_collection(
        self,
        collection: str,
        entries: Iterable[Tuple[str, int, str, np.ndarray]],
    ) -> int:
        """
        Insert all chunks for one collection and build its FAISS index.

        Each entry: `(source_file, section_idx, body, embedding)`.
        The embedding is L2-normalised before insertion so that FAISS
        inner-product search yields cosine similarity.
        """
        faiss = self._faiss()

        materialised: list[tuple[str, int, str, np.ndarray]] = list(entries)
        if not materialised:
            return 0

        vectors = np.stack([e[3] for e in materialised])
        vectors = _normalise(vectors)
        dim = int(vectors.shape[1])

        # Build a fresh FAISS index for this collection.
        idx = faiss.IndexFlatIP(dim)
        idx.add(vectors)
        self._indices[collection] = idx
        self._save_index(collection)

        # Insert metadata rows. `faiss_row` is the row index in the FAISS index.
        rows = [
            (collection, source_file, section_idx, body,
             len(body), dim, faiss_row)
            for faiss_row, (source_file, section_idx, body, _) in enumerate(materialised)
        ]
        self._conn.executemany(
            "INSERT INTO chunks "
            "(collection, source_file, section_idx, body, body_length, "
            " embedding_dim, faiss_row) "
            "VALUES (?, ?, ?, ?, ?, ?, ?);",
            rows,
        )
        self._conn.commit()
        return len(rows)

    # -- reads ------------------------------------------------------------
    def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        """
        Cosine similarity search within one collection.

        Returns a list of `(chunk, score)` in descending score order.
        Empty list if the collection is empty or missing.
        """
        idx = self._load_index(collection)
        if idx is None or idx.ntotal == 0:
            return []

        query = _normalise(np.asarray(query_vector, dtype=np.float32))
        # Guard against a zero query vector — after _normalise it would be
        # zero and searches would return meaningless zero-score neighbours.
        if float(np.linalg.norm(query)) == 0.0:
            return []

        top_k = min(top_k, idx.ntotal)
        scores, faiss_rows = idx.search(query, top_k)
        scores = scores[0]
        faiss_rows = faiss_rows[0]

        # Fetch chunk metadata for the returned FAISS rows.
        placeholders = ",".join("?" for _ in faiss_rows)
        cursor = self._conn.execute(
            f"SELECT id, collection, source_file, section_idx, body, body_length, "
            f"       faiss_row "
            f"FROM chunks WHERE collection = ? AND faiss_row IN ({placeholders});",
            (collection, *[int(r) for r in faiss_rows]),
        )
        row_by_faiss = {row[6]: row for row in cursor.fetchall()}

        out: list[tuple[Chunk, float]] = []
        for score, faiss_row in zip(scores, faiss_rows):
            row = row_by_faiss.get(int(faiss_row))
            if row is None:
                continue
            chunk = Chunk(
                id=row[0], collection=row[1], source_file=row[2],
                section_idx=row[3], body=row[4], body_length=row[5],
                faiss_row=row[6],
            )
            out.append((chunk, float(score)))
        return out

    def size(self, collection: str | None = None) -> int:
        if collection is None:
            return self._conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE collection = ?;",
            (collection,),
        ).fetchone()[0]

    def collection_names(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT collection FROM chunks ORDER BY collection ASC;"
        ).fetchall()
        return [row[0] for row in rows]

    def source_files(self, collection: str | None = None) -> List[str]:
        if collection is None:
            rows = self._conn.execute(
                "SELECT DISTINCT source_file FROM chunks ORDER BY source_file ASC;"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT source_file FROM chunks WHERE collection = ? "
                "ORDER BY source_file ASC;",
                (collection,),
            ).fetchall()
        return [row[0] for row in rows]

    def stats(self, collection: str) -> CollectionStats:
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(body_length), 0), "
            "       COALESCE(AVG(body_length), 0), MAX(embedding_dim) "
            "FROM chunks WHERE collection = ?;",
            (collection,),
        ).fetchone()
        chunk_count, total_chars, avg_chars, dim = row
        return CollectionStats(
            name=collection,
            chunk_count=chunk_count,
            file_count=len(self.source_files(collection)),
            total_chars=total_chars,
            avg_chunk_chars=float(avg_chars),
            embedding_dim=dim,
        )
