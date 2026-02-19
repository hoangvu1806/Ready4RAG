"""
Text chunking strategies for the RAG ingestion pipeline.

The primary strategy is page-based chunking, which treats every PDF page
(delimited by the '\n---\n' marker produced by pdf2md) as one atomic chunk.
An alternative hierarchical strategy splits by headers and then by size.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import ChunkingConfig


@dataclass
class Chunk:
    """A text chunk with full provenance metadata."""

    chunk_id: str
    content: str
    tokens: int
    chunk_index: int
    doc_id: str
    file_path: str
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    # "page" | "text" | "table" | "code" | "equation" | "image"
    content_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "tokens": self.tokens,
            "chunk_index": self.chunk_index,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "page_number": self.page_number,
            "section_header": self.section_header,
            "content_type": self.content_type,
            "metadata": self.metadata,
        }


def _content_hash(content: str, prefix: str = "chunk-") -> str:
    """Return a short hash-based deterministic ID for the given content."""
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}{digest}"


class TextChunker:
    """Markdown-aware text chunker with page-boundary preservation."""

    _PAGE_SEP = re.compile(r"\n---\n")
    _HEADER = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    _TABLE = re.compile(r"\|.+\|[\s\S]*?\n(?=\n|$)")
    _CODE_BLOCK = re.compile(r"```[\s\S]*?```")
    _LATEX_BLOCK = re.compile(r"\$\$[\s\S]*?\$\$")
    _IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self._config = config or ChunkingConfig()

    def chunk_markdown(
        self,
        content: str,
        doc_id: str,
        file_path: str,
        chunk_by_page: Optional[bool] = None,
    ) -> List[Chunk]:
        """
        Split Markdown content into chunks.

        Args:
            content: Full Markdown text (pages separated by '\n---\n').
            doc_id: Stable document identifier.
            file_path: Source file path for metadata.
            chunk_by_page: Override the config flag when provided.

        Returns:
            Ordered list of Chunk objects.
        """
        use_page_strategy = (
            chunk_by_page if chunk_by_page is not None
            else self._config.chunk_by_page
        )
        pages = self._PAGE_SEP.split(content)

        if use_page_strategy:
            return self._chunk_by_page(pages, doc_id, file_path)
        return self._chunk_hierarchical(pages, doc_id, file_path)

    # ------------------------------------------------------------------
    # Page-based strategy
    # ------------------------------------------------------------------

    def _chunk_by_page(
        self, pages: List[str], doc_id: str, file_path: str
    ) -> List[Chunk]:
        chunks = []
        for page_num, raw in enumerate(pages, start=1):
            text = raw.strip()
            if not text:
                continue

            header_match = self._HEADER.search(text)
            section_header = header_match.group(2).strip() if header_match else None

            chunks.append(
                self._make_chunk(
                    content=text,
                    chunk_index=page_num - 1,
                    doc_id=doc_id,
                    file_path=file_path,
                    page_number=page_num,
                    section_header=section_header,
                    content_type="page",
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Hierarchical strategy
    # ------------------------------------------------------------------

    def _chunk_hierarchical(
        self, pages: List[str], doc_id: str, file_path: str
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_index = 0

        for page_num, raw in enumerate(pages, start=1):
            page_chunks = self._split_page(
                raw, page_num, doc_id, file_path, chunk_index
            )
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        return chunks

    def _split_page(
        self,
        page_content: str,
        page_number: int,
        doc_id: str,
        file_path: str,
        start_index: int,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        current_header: Optional[str] = None

        # Protect atomic blocks (tables, code, LaTeX, images) from being split.
        special_blocks = self._extract_special_blocks(page_content)
        processed = page_content
        placeholder_map: Dict[str, Tuple[str, str]] = {}

        for i, (block_type, block_content, start, end) in enumerate(special_blocks):
            ph = f"__SPECIAL_{i}__"
            placeholder_map[ph] = (block_type, block_content)
            processed = processed[:start] + ph + processed[end:]

        sections = self._split_by_headers(processed)
        chunk_index = start_index

        for section_header, section_body in sections:
            if section_header:
                current_header = section_header

            restored = section_body
            for ph, (_, original) in placeholder_map.items():
                restored = restored.replace(ph, original)

            # Detect if the restored content is entirely a special block.
            for ph, (block_type, original) in placeholder_map.items():
                if restored.strip() == original.strip():
                    chunks.append(
                        self._make_chunk(
                            content=original,
                            chunk_index=chunk_index,
                            doc_id=doc_id,
                            file_path=file_path,
                            page_number=page_number,
                            section_header=current_header,
                            content_type=block_type,
                        )
                    )
                    chunk_index += 1
                    break
            else:
                for part in self._split_by_size(restored):
                    if len(part.strip()) >= self._config.min_chunk_size:
                        chunks.append(
                            self._make_chunk(
                                content=part,
                                chunk_index=chunk_index,
                                doc_id=doc_id,
                                file_path=file_path,
                                page_number=page_number,
                                section_header=current_header,
                                content_type="text",
                            )
                        )
                        chunk_index += 1

        return chunks

    def _extract_special_blocks(self, content: str) -> List[Tuple]:
        blocks = []
        for pattern, label in (
            (self._TABLE, "table"),
            (self._CODE_BLOCK, "code"),
            (self._LATEX_BLOCK, "equation"),
            (self._IMAGE_LINK, "image"),
        ):
            for m in pattern.finditer(content):
                blocks.append((label, m.group(), m.start(), m.end()))
        blocks.sort(key=lambda x: x[2])
        return blocks

    def _split_by_headers(self, content: str) -> List[Tuple[Optional[str], str]]:
        sections: List[Tuple[Optional[str], str]] = []
        last_end = 0
        current_header: Optional[str] = None

        for m in self._HEADER.finditer(content):
            if last_end < m.start():
                pre = content[last_end : m.start()].strip()
                if pre:
                    sections.append((current_header, pre))
            current_header = m.group(2).strip()
            last_end = m.end()

        remaining = content[last_end:].strip()
        if remaining:
            sections.append((current_header, remaining))

        if not sections and content.strip():
            sections.append((None, content.strip()))

        return sections

    def _split_by_size(self, content: str) -> List[str]:
        if self._estimate_tokens(content) <= self._config.chunk_size:
            return [content] if content.strip() else []

        for sep in self._config.separators:
            if sep not in content:
                continue

            parts = content.split(sep)
            chunks: List[str] = []
            current = ""

            for part in parts:
                candidate = (current + sep + part) if current else part
                if self._estimate_tokens(candidate) <= self._config.chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part

            if current:
                chunks.append(current)
            return chunks

        # Hard character-count fallback.
        char_limit = self._config.chunk_size * 4
        overlap = self._config.chunk_overlap * 4
        return [
            content[i : i + char_limit]
            for i in range(0, len(content), char_limit - overlap)
        ]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Rough approximation: 4 characters per token.
        return len(text) // 4

    def _make_chunk(
        self,
        content: str,
        chunk_index: int,
        doc_id: str,
        file_path: str,
        page_number: Optional[int],
        section_header: Optional[str],
        content_type: str,
    ) -> Chunk:
        return Chunk(
            chunk_id=_content_hash(content),
            content=content,
            tokens=self._estimate_tokens(content),
            chunk_index=chunk_index,
            doc_id=doc_id,
            file_path=file_path,
            page_number=page_number,
            section_header=section_header,
            content_type=content_type,
            metadata={
                "has_table": content_type == "table",
                "has_code": content_type == "code",
                "has_equation": content_type == "equation",
                "has_image": content_type == "image",
            },
        )
