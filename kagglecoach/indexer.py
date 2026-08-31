"""
Build the three FAISS collections.

    python -m kagglecoach.indexer

For each collection folder listed in `config.toml`'s `[collections]`
section, the indexer:

    1. Finds every `.md` file under `knowledge/<name>/`.
    2. Section-chunks each file.
    3. Batch-embeds every chunk with Foundry Local.
    4. Builds a FAISS index and writes it to `data/faiss/<name>.faiss`.
    5. Inserts chunk metadata into the SQLite table.

Flags:
    --keep      append to existing collections instead of wiping each first
    --collection <name>   rebuild only that collection
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from kagglecoach.chunker import chunk_by_sections
from kagglecoach.settings import SETTINGS, Settings
from kagglecoach.store import MultiCollectionStore


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt"}


def find_source_files(collection_dir: Path) -> List[Tuple[str, str]]:
    """Return sorted (filename, content) pairs for every supported file."""
    collection_dir = Path(collection_dir)
    if not collection_dir.exists():
        raise FileNotFoundError(f"Collection folder not found: {collection_dir}")
    paths = sorted(
        p for p in collection_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No knowledge files in {collection_dir}.")
    return [(p.name, p.read_text(encoding="utf-8")) for p in paths]


def run_indexer(
    settings: Settings = SETTINGS,
    keep_existing: bool = False,
    only_collection: str | None = None,
) -> dict[str, int]:
    """Chunk, embed, and store every collection. Returns per-collection counts."""
    from kagglecoach.models import ModelClient  # lazy so unit tests skip SDK

    client = ModelClient(settings)
    counts: dict[str, int] = {}

    collections_to_build = (
        {only_collection: settings.collections[only_collection]}
        if only_collection is not None
        else settings.collections
    )

    with MultiCollectionStore(settings.db_path, settings.faiss_dir) as store:
        for collection_name, collection_dir in collections_to_build.items():
            print(f"\n[{collection_name}]  scanning {collection_dir}")
            files = find_source_files(collection_dir)

            entries: list[tuple[str, int, str, "np.ndarray"]] = []  # noqa: F821
            for filename, source in files:
                chunks = chunk_by_sections(
                    source,
                    min_chars=settings.chunk_min_chars,
                    max_chars=settings.chunk_max_chars,
                )
                print(f"  {filename:44s} → {len(chunks):3d} chunk(s)")
                if not chunks:
                    continue
                vectors = client.embed(chunks)
                for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
                    entries.append((filename, idx, chunk, vec))

            if not keep_existing:
                store.wipe_collection(collection_name)
            store.add_collection(collection_name, entries)
            counts[collection_name] = len(entries)

        print()
        for name in sorted(counts):
            stats = store.stats(name)
            print(f"[{name}]  {stats.chunk_count} chunk(s) from "
                  f"{stats.file_count} file(s), avg {stats.avg_chunk_chars:.0f} chars, "
                  f"dim {stats.embedding_dim}")

    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KaggleCoach's FAISS collections.")
    parser.add_argument("--keep", action="store_true", help="Do not wipe before indexing.")
    parser.add_argument(
        "--collection",
        default=None,
        help="Rebuild only this collection (tabular / nlp / general_ml).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_indexer(keep_existing=args.keep, only_collection=args.collection)
