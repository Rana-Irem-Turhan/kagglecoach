from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError as exc:
        raise ImportError("Install tomli: pip install tomli") from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"

SYSTEM_PROMPT = """You are KaggleCoach, an experienced ML competition strategist. \
Give concrete, evidence-grounded advice for a specific dataset and user context.
1. Ground every recommendation in the EVIDENCE passages provided.
2. Cite source filenames inline: (source: <filename>)
3. Be concrete — specific model names, hyperparameters, validation strategies.
4. Respect user constraints (compute, deadline, experience).
5. If evidence is insufficient, say so honestly.
"""

@dataclass(frozen=True)
class AzureConfig:
    enabled: bool
    api_version: str

@dataclass(frozen=True)
class Settings:
    project_root: Path
    knowledge_dir: Path
    db_path: Path
    faiss_dir: Path
    chat_model: str
    embedding_model: str
    azure: AzureConfig
    chunk_min_chars: int
    chunk_max_chars: int
    top_k_per_collection: int
    similarity_threshold: float
    min_strong_hits_primary: int
    collections: dict[str, Path]
    temperature: float
    max_tokens: int
    max_questions: int
    high_cardinality_threshold: int
    missing_moderate: float
    missing_severe: float
    size_small_max: int
    size_medium_max: int
    size_large_max: int
    gap_healthy_max: float
    gap_mild_max: float
    rows_per_feature_risky: int
    web_port: int
    system_prompt: str = field(default=SYSTEM_PROMPT)

def _require(data, *keys):
    node = data
    for key in keys:
        if key not in node:
            raise KeyError(f"config.toml missing: {'.'.join(keys)}")
        node = node[key]
    return node

def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "rb") as fp:
        raw = tomllib.load(fp)
    root = config_path.parent
    collections_raw = _require(raw, "collections")
    collections = {name: root / rel for name, rel in collections_raw.items()}
    azure_raw = _require(raw, "models", "azure")
    azure = AzureConfig(
        enabled=bool(azure_raw.get("enabled", False)),
        api_version=str(azure_raw.get("api_version", "2024-08-01-preview")),
    )
    return Settings(
        project_root=root,
        knowledge_dir=root / _require(raw, "paths", "knowledge_dir"),
        db_path=root / _require(raw, "paths", "db_path"),
        faiss_dir=root / _require(raw, "paths", "faiss_dir"),
        chat_model=_require(raw, "models", "chat"),
        embedding_model=_require(raw, "models", "embedding"),
        azure=azure,
        chunk_min_chars=int(_require(raw, "chunking", "min_chars")),
        chunk_max_chars=int(_require(raw, "chunking", "max_chars")),
        top_k_per_collection=int(_require(raw, "retrieval", "top_k_per_collection")),
        similarity_threshold=float(_require(raw, "retrieval", "similarity_threshold")),
        min_strong_hits_primary=int(_require(raw, "retrieval", "min_strong_hits_primary")),
        collections=collections,
        temperature=float(_require(raw, "generation", "temperature")),
        max_tokens=int(_require(raw, "generation", "max_tokens")),
        max_questions=int(_require(raw, "dialogue", "max_questions")),
        high_cardinality_threshold=int(_require(raw, "eda", "high_cardinality_threshold")),
        missing_moderate=float(_require(raw, "eda", "missing_moderate")),
        missing_severe=float(_require(raw, "eda", "missing_severe")),
        size_small_max=int(_require(raw, "eda", "size_small_max")),
        size_medium_max=int(_require(raw, "eda", "size_medium_max")),
        size_large_max=int(_require(raw, "eda", "size_large_max")),
        gap_healthy_max=float(_require(raw, "overfitting", "gap_healthy_max")),
        gap_mild_max=float(_require(raw, "overfitting", "gap_mild_max")),
        rows_per_feature_risky=int(_require(raw, "overfitting", "rows_per_feature_risky")),
        web_port=int(_require(raw, "web", "port")),
    )

SETTINGS = load_settings()