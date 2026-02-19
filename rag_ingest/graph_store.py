"""
Knowledge graph store backed by NetworkX with GraphML persistence.

Entities are stored as directed graph nodes; relationships are edges.
The graph is persisted to disk in GraphML format after each write session.
"""

import json
import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .config import GraphConfig


def _hash_id(content: str, prefix: str = "") -> str:
    """Return a short deterministic ID derived from content."""
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


# Re-export so entity_extractor can import without circular dependency.
compute_hash_id = _hash_id


@dataclass
class Entity:
    """A knowledge graph node representing a named entity."""

    entity_id: str
    entity_name: str
    entity_type: str
    description: str = ""
    source_chunks: List[str] = field(default_factory=list)
    doc_id: str = ""
    file_path: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_chunks": self.source_chunks,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(**data)


@dataclass
class Relationship:
    """A knowledge graph edge representing a directed relationship between entities."""

    relationship_id: str
    source_entity: str  # entity_id
    target_entity: str  # entity_id
    relationship_type: str
    description: str = ""
    weight: float = 1.0
    source_chunk: str = ""
    doc_id: str = ""
    file_path: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relationship_type": self.relationship_type,
            "description": self.description,
            "weight": self.weight,
            "source_chunk": self.source_chunk,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        return cls(**data)


class GraphStoreManager:
    """
    Manages entity/relationship storage in a NetworkX DiGraph.

    The graph is loaded from disk on first access and saved explicitly via
    save(). All public methods are async to integrate with the ingestion
    pipeline's event loop.
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        self._config = config or GraphConfig()
        self._graph: Optional[nx.DiGraph] = None
        self._metadata: Dict[str, Any] = {}
        # name -> entity_id index for O(1) lookup during merge.
        self._name_index: Dict[str, str] = {}

        Path(self._config.storage_path).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def _graph_path(self) -> Path:
        return Path(self._config.storage_path) / self._config.graph_filename

    @property
    def _metadata_path(self) -> Path:
        return Path(self._config.storage_path) / self._config.metadata_filename

    # ------------------------------------------------------------------
    # Internal graph lifecycle
    # ------------------------------------------------------------------

    @property
    def _g(self) -> nx.DiGraph:
        if self._graph is None:
            self._load_or_create()
        return self._graph

    def _load_or_create(self) -> None:
        if self._graph_path.exists():
            try:
                self._graph = nx.read_graphml(str(self._graph_path))
                if self._metadata_path.exists():
                    self._metadata = json.loads(
                        self._metadata_path.read_text(encoding="utf-8")
                    )
                print(
                    f"Loaded graph: {self._graph.number_of_nodes()} nodes, "
                    f"{self._graph.number_of_edges()} edges"
                )
            except Exception as exc:
                print(f"Failed to load graph ({exc}), creating a new one.")
                self._graph = nx.DiGraph()
                self._metadata = {
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
        else:
            self._graph = nx.DiGraph()
            self._metadata = {"created_at": datetime.now(timezone.utc).isoformat()}
            print("Created new knowledge graph.")

        self._rebuild_name_index()

    def _rebuild_name_index(self) -> None:
        self._name_index = {
            data.get("entity_name", node_id): node_id
            for node_id, data in self._g.nodes(data=True)
        }

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Eagerly load the graph so the first write is not delayed."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self._g)
        print(
            f"Graph store ready: {self._g.number_of_nodes()} nodes, "
            f"{self._g.number_of_edges()} edges"
        )

    async def add_entity(self, entity: Entity) -> str:
        """
        Add or merge an entity into the graph.

        If an entity with the same name already exists, its source_chunks list
        is extended and its description is updated if the new one is longer.

        Returns:
            The canonical entity_id used in the graph.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._add_entity_sync(entity))

    def _add_entity_sync(self, entity: Entity) -> str:
        existing_id = self._name_index.get(entity.entity_name)

        if existing_id:
            existing = dict(self._g.nodes[existing_id])
            # Merge source chunks (deduplicate)
            merged_chunks = list(
                set(existing.get("source_chunks", "").split(",") + entity.source_chunks)
            )
            existing["source_chunks"] = ",".join(filter(None, merged_chunks))
            if len(entity.description) > len(existing.get("description", "")):
                existing["description"] = entity.description
            existing["updated_at"] = datetime.now(timezone.utc).isoformat()
            nx.set_node_attributes(self._g, {existing_id: existing})
            return existing_id

        self._g.add_node(
            entity.entity_id,
            entity_name=entity.entity_name,
            entity_type=entity.entity_type,
            description=entity.description,
            source_chunks=",".join(entity.source_chunks),
            doc_id=entity.doc_id,
            file_path=entity.file_path,
            attributes=json.dumps(entity.attributes),
            created_at=entity.created_at,
        )
        self._name_index[entity.entity_name] = entity.entity_id
        return entity.entity_id

    async def add_entities(self, entities: List[Entity]) -> List[str]:
        """Add a batch of entities and return their canonical IDs."""
        ids = []
        for entity in entities:
            ids.append(await self.add_entity(entity))
        return ids

    async def add_relationship(self, relationship: Relationship) -> str:
        """
        Add a relationship edge to the graph.

        If the edge already exists, its weight is incremented.  Placeholder
        nodes are created for any entity_id not yet present in the graph.

        Returns:
            relationship_id
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._add_relationship_sync(relationship)
        )

    def _add_relationship_sync(self, relationship: Relationship) -> str:
        src_id = self._name_index.get(relationship.source_entity, relationship.source_entity)
        tgt_id = self._name_index.get(relationship.target_entity, relationship.target_entity)

        for node_id, name in ((src_id, relationship.source_entity), (tgt_id, relationship.target_entity)):
            if node_id not in self._g:
                self._g.add_node(node_id, entity_name=name, entity_type="unknown")
                self._name_index[name] = node_id

        if self._g.has_edge(src_id, tgt_id):
            self._g[src_id][tgt_id]["weight"] = (
                self._g[src_id][tgt_id].get("weight", 0) + relationship.weight
            )
            self._g[src_id][tgt_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        else:
            self._g.add_edge(
                src_id,
                tgt_id,
                relationship_id=relationship.relationship_id,
                relationship_type=relationship.relationship_type,
                description=relationship.description,
                weight=relationship.weight,
                source_chunk=relationship.source_chunk,
                doc_id=relationship.doc_id,
                file_path=relationship.file_path,
                attributes=json.dumps(relationship.attributes),
                created_at=relationship.created_at,
            )

        return relationship.relationship_id

    async def add_relationships(self, relationships: List[Relationship]) -> List[str]:
        return [await self.add_relationship(r) for r in relationships]

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Look up an entity by ID or by name."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._get_entity_sync(entity_id)
        )

    def _get_entity_sync(self, entity_id: str) -> Optional[Entity]:
        actual_id = self._name_index.get(entity_id, entity_id)
        if actual_id not in self._g:
            return None
        data = dict(self._g.nodes[actual_id])
        return Entity(
            entity_id=actual_id,
            entity_name=data.get("entity_name", ""),
            entity_type=data.get("entity_type", ""),
            description=data.get("description", ""),
            source_chunks=data.get("source_chunks", "").split(","),
            doc_id=data.get("doc_id", ""),
            file_path=data.get("file_path", ""),
            attributes=json.loads(data.get("attributes", "{}")),
            created_at=data.get("created_at", ""),
        )

    async def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relationship_type: Optional[str] = None,
    ) -> List[Tuple[Entity, Relationship]]:
        """
        Return neighboring entities and the connecting relationships.

        Args:
            entity_id: Starting entity (ID or name).
            direction: "out" | "in" | "both"
            relationship_type: Optional filter on relationship_type attribute.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._get_neighbors_sync(entity_id, direction, relationship_type),
        )

    def _get_neighbors_sync(
        self,
        entity_id: str,
        direction: str,
        relationship_type: Optional[str],
    ) -> List[Tuple[Entity, Relationship]]:
        actual_id = self._name_index.get(entity_id, entity_id)
        if actual_id not in self._g:
            return []

        results: List[Tuple[Entity, Relationship]] = []

        def _make_entity(node_id: str) -> Entity:
            d = dict(self._g.nodes[node_id])
            return Entity(
                entity_id=node_id,
                entity_name=d.get("entity_name", ""),
                entity_type=d.get("entity_type", ""),
                description=d.get("description", ""),
                source_chunks=d.get("source_chunks", "").split(","),
                doc_id=d.get("doc_id", ""),
                file_path=d.get("file_path", ""),
            )

        if direction in ("out", "both"):
            for nbr in self._g.successors(actual_id):
                edge = self._g[actual_id][nbr]
                if relationship_type and edge.get("relationship_type") != relationship_type:
                    continue
                results.append((
                    _make_entity(nbr),
                    Relationship(
                        relationship_id=edge.get("relationship_id", ""),
                        source_entity=actual_id,
                        target_entity=nbr,
                        relationship_type=edge.get("relationship_type", ""),
                        description=edge.get("description", ""),
                        weight=edge.get("weight", 1.0),
                    ),
                ))

        if direction in ("in", "both"):
            for nbr in self._g.predecessors(actual_id):
                edge = self._g[nbr][actual_id]
                if relationship_type and edge.get("relationship_type") != relationship_type:
                    continue
                results.append((
                    _make_entity(nbr),
                    Relationship(
                        relationship_id=edge.get("relationship_id", ""),
                        source_entity=nbr,
                        target_entity=actual_id,
                        relationship_type=edge.get("relationship_type", ""),
                        description=edge.get("description", ""),
                        weight=edge.get("weight", 1.0),
                    ),
                ))

        return results

    async def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: [
                Entity(
                    entity_id=nid,
                    entity_name=d.get("entity_name", ""),
                    entity_type=entity_type,
                    description=d.get("description", ""),
                    source_chunks=d.get("source_chunks", "").split(","),
                    doc_id=d.get("doc_id", ""),
                    file_path=d.get("file_path", ""),
                )
                for nid, d in self._g.nodes(data=True)
                if d.get("entity_type") == entity_type
            ],
        )

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove all nodes that originated from the given document."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._delete_by_doc_id_sync(doc_id)
        )

    def _delete_by_doc_id_sync(self, doc_id: str) -> int:
        to_remove = [
            nid
            for nid, d in self._g.nodes(data=True)
            if d.get("doc_id") == doc_id
        ]
        for nid in to_remove:
            name = self._g.nodes[nid].get("entity_name")
            self._name_index.pop(name, None)
            self._g.remove_node(nid)
        return len(to_remove)

    async def save(self) -> None:
        """Persist the graph and metadata to disk."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_sync)

    def _save_sync(self) -> None:
        nx.write_graphml(self._g, str(self._graph_path))
        self._metadata.update(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "nodes_count": self._g.number_of_nodes(),
                "edges_count": self._g.number_of_edges(),
            }
        )
        self._metadata_path.write_text(
            json.dumps(self._metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"Graph saved: {self._g.number_of_nodes()} nodes, "
            f"{self._g.number_of_edges()} edges"
        )

    async def get_stats(self) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._stats_sync)

    def _stats_sync(self) -> Dict[str, Any]:
        entity_types: Dict[str, int] = {}
        for _, d in self._g.nodes(data=True):
            t = d.get("entity_type", "unknown")
            entity_types[t] = entity_types.get(t, 0) + 1

        relationship_types: Dict[str, int] = {}
        for _, _, d in self._g.edges(data=True):
            t = d.get("relationship_type", "unknown")
            relationship_types[t] = relationship_types.get(t, 0) + 1

        return {
            "nodes_count": self._g.number_of_nodes(),
            "edges_count": self._g.number_of_edges(),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "is_connected": (
                nx.is_weakly_connected(self._g)
                if self._g.number_of_nodes() > 0
                else True
            ),
            "density": nx.density(self._g),
        }
