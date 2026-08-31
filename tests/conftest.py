"""Shared pytest fixtures for KaggleCoach tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from kagglecoach.settings import SETTINGS, Settings
from kagglecoach.store import MultiCollectionStore


class FakeModelClient:
    """
    Deterministic stand-in for `ModelClient` — no Foundry Local, no Azure.

    * Embeddings are computed as normalised character histograms so the
      similarity structure is meaningful for retrieval tests.
    * Chat replies echo back the retrieved evidence so downstream tests
      can verify grounding without a real LLM.
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self.use_azure_for_chat = False
        self.chat_calls: list[tuple[str, str]] = []

    @property
    def active_mode(self) -> str:
        return "azure" if self.use_azure_for_chat else "local"

    def prewarm(self) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)

    def chat(self, system: str, user: str) -> str:
        self.chat_calls.append((system, user))
        # Echo the citation-worthy portion so grounding tests can pattern-match.
        return f"[fake response] using evidence from user prompt of length {len(user)}"

    def _embed_one(self, text: str) -> np.ndarray:
        """Character-histogram embedding — deterministic and similarity-sensitive."""
        vec = np.zeros(self.dim, dtype=np.float32)
        for ch in text.lower():
            vec[ord(ch) % self.dim] += 1.0
        # L2 normalise so cosine similarity is well-behaved.
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec


@pytest.fixture()
def fake_model() -> FakeModelClient:
    return FakeModelClient()


@pytest.fixture()
def tmp_store(tmp_path: Path):
    """Fresh MultiCollectionStore under a temp directory."""
    with MultiCollectionStore(
        db_path=tmp_path / "test.sqlite",
        faiss_dir=tmp_path / "faiss",
    ) as store:
        yield store


@pytest.fixture()
def settings() -> Settings:
    """The canonical settings — read-only for tests."""
    return SETTINGS
