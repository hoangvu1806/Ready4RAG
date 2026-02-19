"""
Configuration for the RAG ingestion pipeline.

All values are resolved from environment variables or a .env file located at
the project root.  Sub-configurations are composed into RAGIngestConfig, which
is the single entry point consumed by all pipeline components.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default, cast: type = str):
    value = os.getenv(key, default)
    if value is None:
        return default
    if cast is bool:
        return str(value).lower() in ("true", "1", "yes")
    return cast(value)


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------


@dataclass
class QdrantConfig:
    """Qdrant vector database connection settings."""

    use_local: bool = field(
        default_factory=lambda: _env("QDRANT_USE_LOCAL", True, bool)
    )
    # Path used when use_local=True (embedded Qdrant).
    path: str = field(
        default_factory=lambda: _env("QDRANT_PATH", "./database/vector")
    )
    # Host/port used when use_local=False (remote Qdrant server).
    host: str = field(default_factory=lambda: _env("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: _env("QDRANT_PORT", 6333, int))

    # Collection names for the four semantic views of the data.
    chunks_collection: str = "chunks"
    entities_collection: str = "entities"
    relationships_collection: str = "relationships"
    summaries_collection: str = "summaries"


@dataclass
class GraphConfig:
    """NetworkX knowledge graph persistence settings."""

    storage_path: str = field(
        default_factory=lambda: _env("GRAPH_STORAGE_PATH", "./database/graph")
    )
    graph_filename: str = "knowledge_graph.graphml"
    metadata_filename: str = "graph_metadata.json"


@dataclass
class EmbeddingConfig:
    """Embedding model settings."""

    # Provider: openai | gemini | local
    provider: str = field(
        default_factory=lambda: _env("EMBEDDING_PROVIDER", "gemini")
    )

    # OpenAI
    openai_api_key: str = field(
        default_factory=lambda: _env("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    # text-embedding-3-small: 1536 dim | text-embedding-3-large: 3072 dim
    openai_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Google Gemini
    gemini_api_key: str = field(
        default_factory=lambda: _env("GEMINI_API_KEY", "")
    )
    # gemini-embedding-001: 3072 dim (supports MRL truncation to 768 / 1536 / 3072)
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    )

    # Local sentence-transformers
    local_model: str = field(
        default_factory=lambda: _env(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # Explicit dimension override; set to match the selected model.
    embedding_dim: int = field(
        default_factory=lambda: _env("EMBEDDING_DIM", 3072, int)
    )
    batch_size: int = field(
        default_factory=lambda: _env("EMBEDDING_BATCH_SIZE", 32, int)
    )


@dataclass
class LLMConfig:
    """LLM settings for entity extraction and summary generation."""

    # Provider: openai | gemini | groq
    provider: str = field(
        default_factory=lambda: _env("LLM_PROVIDER", "gemini")
    )

    # OpenAI
    openai_api_key: str = field(
        default_factory=lambda: _env("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    openai_model: str = field(
        default_factory=lambda: _env("OPENAI_LLM_MODEL", "gpt-4o-mini")
    )

    # Google Gemini
    gemini_api_key: str = field(
        default_factory=lambda: _env("GEMINI_API_KEY", "")
    )
    # gemini-2.0-flash: stable GA release (February 2025).
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash")
    )

    # Groq
    groq_api_key: str = field(
        default_factory=lambda: _env("GROQ_API_KEY", "")
    )
    # llama-3.3-70b-versatile: recommended general text model on Groq.
    groq_model: str = field(
        default_factory=lambda: _env("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    )

    max_tokens: int = field(
        default_factory=lambda: _env("LLM_MAX_TOKENS", 4096, int)
    )
    temperature: float = field(
        default_factory=lambda: _env("LLM_TEMPERATURE", 0.0, float)
    )


@dataclass
class ChunkingConfig:
    """Text chunking strategy settings."""

    # When True, one chunk is produced per PDF page (recommended for structured docs).
    chunk_by_page: bool = field(
        default_factory=lambda: _env("CHUNK_BY_PAGE", True, bool)
    )

    # Size-based settings used when chunk_by_page=False.
    chunk_size: int = field(
        default_factory=lambda: _env("CHUNK_SIZE", 512, int)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _env("CHUNK_OVERLAP", 64, int)
    )
    min_chunk_size: int = field(
        default_factory=lambda: _env("MIN_CHUNK_SIZE", 100, int)
    )

    # Separator priority order for hierarchical splitting.
    separators: List[str] = field(
        default_factory=lambda: [
            "\n---\n",   # Page separator inserted by pdf2md
            "\n## ",     # H2 header
            "\n### ",    # H3 header
            "\n#### ",   # H4 header
            "\n\n",      # Paragraph break
            "\n",        # Line break
            ". ",        # Sentence boundary
            " ",         # Word boundary
        ]
    )


# ---------------------------------------------------------------------------
# Root configuration
# ---------------------------------------------------------------------------


@dataclass
class RAGIngestConfig:
    """Top-level configuration for the RAG ingestion pipeline."""

    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

    input_dir: str = field(
        default_factory=lambda: _env("RAG_INPUT_DIR", "./output/markdown")
    )
    max_workers: int = field(
        default_factory=lambda: _env("RAG_MAX_WORKERS", 4, int)
    )
    enable_entity_extraction: bool = field(
        default_factory=lambda: _env("RAG_ENABLE_ENTITY_EXTRACTION", True, bool)
    )
    enable_relationship_extraction: bool = field(
        default_factory=lambda: _env("RAG_ENABLE_RELATIONSHIP_EXTRACTION", True, bool)
    )
    enable_summary_generation: bool = field(
        default_factory=lambda: _env("RAG_ENABLE_SUMMARY", True, bool)
    )

    def validate(self) -> None:
        """Raise ValueError for any invalid or missing required setting."""
        if self.embedding.provider == "openai" and not self.embedding.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        if self.embedding.provider == "gemini" and not self.embedding.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for EMBEDDING_PROVIDER=gemini")
        if self.llm.provider == "openai" and not self.llm.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        if self.llm.provider == "gemini" and not self.llm.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for LLM_PROVIDER=gemini")
        if self.llm.provider == "groq" and not self.llm.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for LLM_PROVIDER=groq")

    def ensure_directories(self) -> None:
        """Create runtime directories that must exist before the pipeline starts."""
        Path(self.qdrant.path).mkdir(parents=True, exist_ok=True)
        Path(self.graph.storage_path).mkdir(parents=True, exist_ok=True)
