"""
Qdrant vector store manager for the RAG ingestion pipeline.

Manages four named collections:
- chunks       : page or section text chunks
- entities     : named entity embeddings
- relationships: relationship embeddings
- summaries    : document and section summary embeddings
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .chunker import Chunk
from .config import QdrantConfig


def _stable_point_id(value: str) -> int:
    """Convert a string ID to a deterministic non-negative integer for Qdrant."""
    digest = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2 ** 63)


class SearchResult:
    """A single vector search hit."""

    __slots__ = ("id", "score", "payload")

    def __init__(self, id: str, score: float, payload: Dict[str, Any]):
        self.id = id
        self.score = score
        self.payload = payload


class VectorStoreManager:
    """Manages CRUD operations across all four Qdrant collections."""

    def __init__(
        self,
        config: Optional[QdrantConfig] = None,
        embedding_dim: int = 3072,
    ):
        self._config = config or QdrantConfig()
        self._embedding_dim = embedding_dim
        self._client: Optional[QdrantClient] = None

        self._collections = {
            "chunks": self._config.chunks_collection,
            "entities": self._config.entities_collection,
            "relationships": self._config.relationships_collection,
            "summaries": self._config.summaries_collection,
        }

    @property
    def _lazy_client(self) -> QdrantClient:
        if self._client is None:
            if self._config.use_local:
                storage = Path(self._config.path)
                storage.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(storage))
            else:
                self._client = QdrantClient(
                    host=self._config.host,
                    port=self._config.port,
                )
        return self._client

    async def initialize(self) -> None:
        """Ensure all four collections exist (create them if necessary)."""
        for collection_name in self._collections.values():
            await self._ensure_collection(collection_name)
        print(f"Vector store ready: {len(self._collections)} collections")

    async def _ensure_collection(self, name: str) -> None:
        loop = asyncio.get_event_loop()
        try:
            exists = await loop.run_in_executor(
                None, lambda: self._lazy_client.collection_exists(name)
            )
            if not exists:
                await loop.run_in_executor(
                    None,
                    lambda n=name: self._lazy_client.create_collection(
                        collection_name=n,
                        vectors_config=VectorParams(
                            size=self._embedding_dim,
                            distance=Distance.COSINE,
                        ),
                    ),
                )
                print(f"Created collection: {name}")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize collection '{name}': {exc}") from exc

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert_chunks(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> None:
        points = [
            PointStruct(
                id=_stable_point_id(chunk.chunk_id),
                vector=emb,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "tokens": chunk.tokens,
                    "chunk_index": chunk.chunk_index,
                    "doc_id": chunk.doc_id,
                    "file_path": chunk.file_path,
                    "page_number": chunk.page_number,
                    "section_header": chunk.section_header,
                    "content_type": chunk.content_type,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        await self._upsert_points(self._collections["chunks"], points)

    async def upsert_entities(
        self, entities: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        points = [
            PointStruct(
                id=_stable_point_id(entity["entity_id"]),
                vector=emb,
                payload={
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "entity_type": entity["entity_type"],
                    "description": entity.get("description", ""),
                    "source_chunks": entity.get("source_chunks", []),
                    "doc_id": entity.get("doc_id", ""),
                    "file_path": entity.get("file_path", ""),
                    "attributes": entity.get("attributes", {}),
                },
            )
            for entity, emb in zip(entities, embeddings)
        ]
        await self._upsert_points(self._collections["entities"], points)

    async def upsert_relationships(
        self, relationships: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        points = [
            PointStruct(
                id=_stable_point_id(rel["relationship_id"]),
                vector=emb,
                payload={
                    "relationship_id": rel["relationship_id"],
                    "source_entity": rel["source_entity"],
                    "target_entity": rel["target_entity"],
                    "relationship_type": rel["relationship_type"],
                    "description": rel.get("description", ""),
                    "weight": rel.get("weight", 1.0),
                    "source_chunk": rel.get("source_chunk", ""),
                    "doc_id": rel.get("doc_id", ""),
                    "file_path": rel.get("file_path", ""),
                },
            )
            for rel, emb in zip(relationships, embeddings)
        ]
        await self._upsert_points(self._collections["relationships"], points)

    async def upsert_summaries(
        self, summaries: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        points = [
            PointStruct(
                id=_stable_point_id(summary["summary_id"]),
                vector=emb,
                payload={
                    "summary_id": summary["summary_id"],
                    "content": summary["content"],
                    "summary_type": summary.get("summary_type", "document"),
                    "doc_id": summary.get("doc_id", ""),
                    "file_path": summary.get("file_path", ""),
                    "section_header": summary.get("section_header"),
                    "chunk_ids": summary.get("chunk_ids", []),
                },
            )
            for summary, emb in zip(summaries, embeddings)
        ]
        await self._upsert_points(self._collections["summaries"], points)

    async def _upsert_points(
        self, collection_name: str, points: List[PointStruct]
    ) -> None:
        if not points:
            return
        loop = asyncio.get_event_loop()
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await loop.run_in_executor(
                None,
                lambda b=batch, cn=collection_name: self._lazy_client.upsert(
                    collection_name=cn, points=b
                ),
            )

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    async def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        doc_id: Optional[str] = None,
        content_type: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        conditions = []
        if doc_id:
            conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if content_type:
            conditions.append(
                FieldCondition(key="content_type", match=MatchValue(value=content_type))
            )
        query_filter = Filter(must=conditions) if conditions else None

        return await self._search(
            self._collections["chunks"], query_embedding, top_k, query_filter, score_threshold
        )

    async def search_entities(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        entity_type: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        conditions = []
        if entity_type:
            conditions.append(
                FieldCondition(key="entity_type", match=MatchValue(value=entity_type))
            )
        query_filter = Filter(must=conditions) if conditions else None

        return await self._search(
            self._collections["entities"], query_embedding, top_k, query_filter, score_threshold
        )

    async def search_relationships(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        return await self._search(
            self._collections["relationships"], query_embedding, top_k, None, score_threshold
        )

    async def search_summaries(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        summary_type: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> List[SearchResult]:
        conditions = []
        if summary_type:
            conditions.append(
                FieldCondition(key="summary_type", match=MatchValue(value=summary_type))
            )
        query_filter = Filter(must=conditions) if conditions else None

        return await self._search(
            self._collections["summaries"], query_embedding, top_k, query_filter, score_threshold
        )

    async def _search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int,
        query_filter: Optional[Filter],
        score_threshold: float,
    ) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        hits = await loop.run_in_executor(
            None,
            lambda: self._lazy_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            ),
        )
        return [
            SearchResult(id=str(h.id), score=h.score, payload=h.payload)
            for h in hits
        ]

    # ------------------------------------------------------------------
    # Delete / introspection
    # ------------------------------------------------------------------

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove all vectors belonging to the given document from all collections."""
        loop = asyncio.get_event_loop()
        for name in self._collections.values():
            try:
                await loop.run_in_executor(
                    None,
                    lambda cn=name: self._lazy_client.delete(
                        collection_name=cn,
                        points_selector=models.FilterSelector(
                            filter=Filter(
                                must=[
                                    FieldCondition(
                                        key="doc_id",
                                        match=MatchValue(value=doc_id),
                                    )
                                ]
                            )
                        ),
                    ),
                )
            except Exception as exc:
                print(f"Warning: could not delete from '{name}': {exc}")

    async def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        stats: Dict[str, Dict[str, Any]] = {}
        for alias, name in self._collections.items():
            try:
                info = await loop.run_in_executor(
                    None, lambda cn=name: self._lazy_client.get_collection(cn)
                )
                stats[alias] = {
                    "name": name,
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status.name,
                }
            except Exception as exc:
                stats[alias] = {"error": str(exc)}
        return stats

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
