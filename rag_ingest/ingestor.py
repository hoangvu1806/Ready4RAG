"""
RAG ingestion pipeline orchestrator.

Reads Markdown files produced by pdf2md, chunks the content, generates
embeddings, extracts entities and relationships, and persists everything to
both the vector store (Qdrant) and the knowledge graph (NetworkX / GraphML).
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunker import Chunk, TextChunker
from .config import RAGIngestConfig
from .embeddings import EmbeddingProvider
from .entity_extractor import EntityExtractor, ExtractionResult
from .graph_store import GraphStoreManager
from .vector_store import VectorStoreManager


@dataclass
class IngestionResult:
    """Outcome of ingesting a single Markdown document."""

    file_path: str
    doc_id: str
    chunks_count: int = 0
    entities_count: int = 0
    relationships_count: int = 0
    summaries_count: int = 0
    processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


class RAGIngestor:
    """
    Orchestrates the full ingestion pipeline for Markdown documents.

    Lifecycle:
        ingestor = RAGIngestor(config)
        await ingestor.initialize()
        result = await ingestor.ingest_markdown("path/to/doc.md")
        ingestor.close()
    """

    def __init__(self, config: Optional[RAGIngestConfig] = None):
        self._config = config or RAGIngestConfig()
        self._config.validate()
        self._config.ensure_directories()

        self._chunker = TextChunker(self._config.chunking)
        self._embedding_provider = EmbeddingProvider(self._config.embedding)

        self.vector_store = VectorStoreManager(
            config=self._config.qdrant,
            embedding_dim=self._embedding_provider.dimension,
        )
        self.graph_store = GraphStoreManager(self._config.graph)
        self._extractor = EntityExtractor(self._config.llm)

        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize all storage backends (idempotent)."""
        if self._is_initialized:
            return
        await self.vector_store.initialize()
        await self.graph_store.initialize()
        self._is_initialized = True
        print("RAG ingestor initialized.")

    # ------------------------------------------------------------------
    # Main ingestion entry point
    # ------------------------------------------------------------------

    async def ingest_markdown(
        self,
        markdown_path: str,
        doc_id: Optional[str] = None,
        skip_if_exists: bool = True,
    ) -> IngestionResult:
        """
        Ingest a single Markdown file into the pipeline.

        Args:
            markdown_path: Path to the Markdown file.
            doc_id: Stable document identifier. Derived from path when None.
            skip_if_exists: When True, skip re-ingesting already-stored chunks.

        Returns:
            IngestionResult describing what was stored.
        """
        await self.initialize()

        src = Path(markdown_path)
        if not src.exists():
            return IngestionResult(
                file_path=str(src),
                doc_id=doc_id or "",
                success=False,
                error=f"File not found: {src}",
            )

        if doc_id is None:
            doc_id = hashlib.md5(str(src.resolve()).encode()).hexdigest()[:16]

        start = time.monotonic()
        print(f"Ingesting: {src.name} (doc_id={doc_id})")

        try:
            content = src.read_text(encoding="utf-8")
            chunks = self._chunker.chunk_markdown(
                content=content,
                doc_id=doc_id,
                file_path=str(src),
            )

            if not chunks:
                return IngestionResult(
                    file_path=str(src),
                    doc_id=doc_id,
                    success=True,
                    processing_time=time.monotonic() - start,
                )

            result = await self._process_chunks(
                chunks, doc_id, str(src), skip_if_exists
            )
            await self.graph_store.save()
            result.processing_time = time.monotonic() - start
            return result

        except Exception as exc:
            return IngestionResult(
                file_path=str(src),
                doc_id=doc_id,
                success=False,
                error=str(exc),
                processing_time=time.monotonic() - start,
            )

    # ------------------------------------------------------------------
    # Directory ingestion
    # ------------------------------------------------------------------

    async def ingest_directory(
        self,
        directory: str,
        pattern: str = "*.md",
        recursive: bool = True,
    ) -> List[IngestionResult]:
        """Ingest all matching Markdown files in a directory."""
        await self.initialize()

        base = Path(directory)
        if not base.exists():
            raise FileNotFoundError(f"Directory not found: {base}")

        files = list(base.rglob(pattern) if recursive else base.glob(pattern))
        if not files:
            print(f"No files matching '{pattern}' found in {base}")
            return []

        print(f"Ingesting {len(files)} file(s) from {base} ...")
        semaphore = asyncio.Semaphore(self._config.max_workers)

        async def _guarded(fp: Path) -> IngestionResult:
            async with semaphore:
                return await self.ingest_markdown(str(fp))

        results = await asyncio.gather(*(_guarded(f) for f in files))
        return list(results)

    # ------------------------------------------------------------------
    # Document deletion
    # ------------------------------------------------------------------

    async def delete_document(self, doc_id: str) -> bool:
        """Remove all data associated with a document from vector and graph stores."""
        await self.initialize()
        try:
            await self.vector_store.delete_by_doc_id(doc_id)
            removed = await self.graph_store.delete_by_doc_id(doc_id)
            await self.graph_store.save()
            print(f"Deleted doc_id={doc_id}: {removed} graph nodes removed.")
            return True
        except Exception as exc:
            print(f"Delete failed for doc_id={doc_id}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        await self.initialize()
        return {
            "vector_store": await self.vector_store.get_collection_stats(),
            "graph_store": await self.graph_store.get_stats(),
        }

    def close(self) -> None:
        self.vector_store.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _process_chunks(
        self,
        chunks: List[Chunk],
        doc_id: str,
        file_path: str,
        skip_if_exists: bool,
    ) -> IngestionResult:
        result = IngestionResult(
            file_path=file_path,
            doc_id=doc_id,
            chunks_count=len(chunks),
        )

        # Embed all chunks in one batched call.
        chunk_texts = [c.content for c in chunks]
        chunk_embeddings = await self._embedding_provider.embed_texts(chunk_texts)
        await self.vector_store.upsert_chunks(chunks, chunk_embeddings)

        all_entities: List = []
        all_relationships: List = []
        all_summaries: List[Dict] = []

        if not self._config.enable_entity_extraction:
            return result

        # Process chunks concurrently but bounded by a semaphore.
        semaphore = asyncio.Semaphore(self._config.max_workers)

        async def _extract_chunk(chunk: Chunk) -> ExtractionResult:
            async with semaphore:
                return await self._extractor.extract(chunk)

        extraction_results = await asyncio.gather(
            *(_extract_chunk(c) for c in chunks)
        )

        for chunk, extraction in zip(chunks, extraction_results):
            all_entities.extend(extraction.entities)
            if self._config.enable_relationship_extraction:
                all_relationships.extend(extraction.relationships)

            if self._config.enable_summary_generation and extraction.summary:
                all_summaries.append(
                    self._build_summary_record(chunk, extraction.summary, doc_id, file_path)
                )

        # Graph persistence
        entity_ids = await self.graph_store.add_entities(all_entities)
        if self._config.enable_relationship_extraction:
            await self.graph_store.add_relationships(all_relationships)

        # Vector persistence for entities and relationships
        if all_entities:
            entity_texts = [
                f"{e.entity_name}: {e.description}" for e in all_entities
            ]
            entity_embeddings = await self._embedding_provider.embed_texts(entity_texts)
            entity_dicts = [e.to_dict() for e in all_entities]
            for ed, eid in zip(entity_dicts, entity_ids):
                ed["entity_id"] = eid
            await self.vector_store.upsert_entities(entity_dicts, entity_embeddings)

        if self._config.enable_relationship_extraction and all_relationships:
            rel_texts = [
                f"{r.source_entity} {r.relationship_type} {r.target_entity}: {r.description}"
                for r in all_relationships
            ]
            rel_embeddings = await self._embedding_provider.embed_texts(rel_texts)
            await self.vector_store.upsert_relationships(
                [r.to_dict() for r in all_relationships], rel_embeddings
            )

        if all_summaries:
            summary_texts = [s["content"] for s in all_summaries]
            summary_embeddings = await self._embedding_provider.embed_texts(summary_texts)
            await self.vector_store.upsert_summaries(all_summaries, summary_embeddings)

        result.entities_count = len(all_entities)
        result.relationships_count = len(all_relationships)
        result.summaries_count = len(all_summaries)

        print(
            f"Processed {file_path}: "
            f"{len(chunks)} chunks, {len(all_entities)} entities, "
            f"{len(all_relationships)} relationships"
        )
        return result

    @staticmethod
    def _build_summary_record(
        chunk: Chunk, summary_text: str, doc_id: str, file_path: str
    ) -> Dict[str, Any]:
        summary_id = hashlib.md5(
            f"summary:{chunk.chunk_id}".encode()
        ).hexdigest()[:16]
        return {
            "summary_id": summary_id,
            "content": summary_text,
            "summary_type": "page" if chunk.content_type == "page" else "section",
            "doc_id": doc_id,
            "file_path": file_path,
            "section_header": chunk.section_header,
            "chunk_ids": [chunk.chunk_id],
        }
