"""
Interactive RAG chatbot.

Retrieves context from both the Qdrant vector store and the NetworkX knowledge
graph, then generates a grounded answer using the configured LLM.

Usage:
    python chatbot.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running from the project root without installing as a package.
sys.path.insert(0, str(Path(__file__).parent))

from rag_ingest.config import RAGIngestConfig
from rag_ingest.embeddings import EmbeddingProvider
from rag_ingest.entity_extractor import LLMClient
from rag_ingest.graph_store import GraphStoreManager
from rag_ingest.vector_store import VectorStoreManager


_SYSTEM_PROMPT = """\
You are a precise knowledge assistant. Answer the user's question using ONLY
the provided context. If the context does not contain sufficient information,
say so directly. Do not invent facts.

Context:
{context}

Question: {question}

Answer:"""


class RAGChatbot:
    def __init__(self, config: Optional[RAGIngestConfig] = None):
        self._config = config or RAGIngestConfig()
        self._config.validate()

        self._embedding = EmbeddingProvider(self._config.embedding)
        self._vector_store = VectorStoreManager(
            config=self._config.qdrant,
            embedding_dim=self._embedding.dimension,
        )
        self._graph_store = GraphStoreManager(self._config.graph)
        print(self._config.llm)
        self._llm = LLMClient(self._config.llm)
        self._history: List[Dict[str, str]] = []
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._vector_store.initialize()
        await self._graph_store.initialize()
        self._initialized = True
        print("Chatbot ready.")

    async def chat(self, question: str, verbose: bool = False) -> str:
        await self.initialize()

        context = await self._retrieve_context(question)
        if verbose:
            print("\n" + "=" * 60)
            print("RETRIEVED CONTEXT:")
            print("-" * 60)
            print(context)
            print("=" * 60 + "\n")

        if not context:
            return (
                "No relevant information was found in the knowledge base "
                "for your question."
            )

        prompt = _SYSTEM_PROMPT.format(context=context, question=question)
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None, lambda: self._llm.complete(prompt)
        )

        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})
        return answer

    def close(self) -> None:
        self._vector_store.close()

    async def _retrieve_context(self, query: str) -> str:
        query_embedding = await self._embedding.embed_text(query)

        # Parallel vector search across all three semantic views.
        chunk_hits, entity_hits, rel_hits = await asyncio.gather(
            self._vector_store.search_chunks(query_embedding, top_k=5),
            self._vector_store.search_entities(query_embedding, top_k=3),
            self._vector_store.search_relationships(query_embedding, top_k=3),
        )

        context_parts: List[str] = []

        # Relevant text passages (highest priority)
        for hit in chunk_hits:
            content = hit.payload.get("content", "").strip()
            page = hit.payload.get("page_number")
            header = hit.payload.get("section_header", "")
            label = f"[Page {page}]" if page else ""
            if header:
                label = f"{label} [{header}]"
            context_parts.append(f"PASSAGE {label}:\n{content}")

        # Entity descriptions
        if entity_hits:
            entity_lines = []
            for hit in entity_hits:
                name = hit.payload.get("entity_name", "")
                desc = hit.payload.get("description", "")
                etype = hit.payload.get("entity_type", "")
                if name:
                    entity_lines.append(f"  {name} ({etype}): {desc}")
            if entity_lines:
                context_parts.append("ENTITIES:\n" + "\n".join(entity_lines))

            # Graph traversal: expand top entity by one hop.
            top_entity = entity_hits[0].payload.get("entity_name")
            if top_entity:
                neighbors = await self._graph_store.get_neighbors(
                    top_entity, direction="both"
                )
                if neighbors:
                    neighbor_lines = []
                    for nb_entity, nb_rel in neighbors[:5]:
                        neighbor_lines.append(
                            f"  {top_entity} --[{nb_rel.relationship_type}]--> "
                            f"{nb_entity.entity_name}: {nb_rel.description}"
                        )
                    context_parts.append(
                        "GRAPH NEIGHBORS:\n" + "\n".join(neighbor_lines)
                    )

        # Relationship descriptions
        if rel_hits:
            rel_lines = []
            for hit in rel_hits:
                src = hit.payload.get("source_entity", "")
                rel_type = hit.payload.get("relationship_type", "")
                tgt = hit.payload.get("target_entity", "")
                desc = hit.payload.get("description", "")
                rel_lines.append(f"  {src} --[{rel_type}]--> {tgt}: {desc}")
            if rel_lines:
                context_parts.append("RELATIONSHIPS:\n" + "\n".join(rel_lines))

        return "\n\n".join(context_parts)


async def _interactive_loop(chatbot: RAGChatbot) -> None:
    print("RAG Chatbot - type 'exit' to quit, '/verbose' to toggle context display.")
    print("-" * 60)
    verbose_mode = False

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if question.lower() in ("exit", "quit", "q"):
            break
        if question.lower() == "/verbose":
            verbose_mode = not verbose_mode
            print(f"Verbose mode: {'ON' if verbose_mode else 'OFF'}")
            continue
        if not question:
            continue

        answer = await chatbot.chat(question, verbose=verbose_mode)
        print(f"\nAssistant:\n{answer}")


async def main() -> None:
    config = RAGIngestConfig()
    chatbot = RAGChatbot(config)

    try:
        await _interactive_loop(chatbot)
    finally:
        chatbot.close()


if __name__ == "__main__":
    asyncio.run(main())
