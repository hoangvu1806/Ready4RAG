"""
rag_ingest - Hybrid RAG ingestion pipeline (Vector + Graph).
"""

from .config import RAGIngestConfig
from .ingestor import RAGIngestor
from .vector_store import VectorStoreManager
from .graph_store import GraphStoreManager
from .chunker import TextChunker
from .embeddings import EmbeddingProvider

__all__ = [
    "RAGIngestConfig",
    "RAGIngestor",
    "VectorStoreManager",
    "GraphStoreManager",
    "TextChunker",
    "EmbeddingProvider",
]

__version__ = "2.0.0"
