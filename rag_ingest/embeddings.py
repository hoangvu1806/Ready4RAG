"""
Embedding providers for the RAG ingestion pipeline.

Each provider implements BaseEmbeddingProvider. The public EmbeddingProvider
class acts as a factory that selects the correct implementation from config.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from .config import EmbeddingConfig


class BaseEmbeddingProvider(ABC):
    """Interface for text embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Return the embedding vector for a single text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality produced by this provider."""


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using AsyncOpenAI client."""

    # Known dimensions for OpenAI embedding models.
    _DIMENSION_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: EmbeddingConfig):
        self._config = config
        self._client = None
        self._dimension = self._DIMENSION_MAP.get(config.openai_model, 1536)

    @property
    def _lazy_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._config.openai_api_key,
                base_url=self._config.openai_base_url,
            )
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> List[float]:
        result = await self.embed_texts([text])
        return result[0]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = self._config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._lazy_client.embeddings.create(
                model=self._config.openai_model,
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider.

    Uses the synchronous google-generativeai SDK wrapped in an executor
    because the SDK does not expose an async interface for embeddings.
    """

    # Known dimensions for Gemini embedding models.
    _DIMENSION_MAP = {
        "models/text-embedding-004": 768,
        "models/embedding-001": 768,
        "gemini-embedding-001": 3072,
        "models/gemini-embedding-001": 3072,
    }

    def __init__(self, config: EmbeddingConfig):
        self._config = config
        self._genai = None
        self._dimension = self._DIMENSION_MAP.get(config.gemini_model, 3072)

    @property
    def _client(self):
        if self._genai is None:
            import google.generativeai as genai
            genai.configure(api_key=self._config.gemini_api_key)
            self._genai = genai
        return self._genai

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._client.embed_content(
                model=self._config.gemini_model,
                content=text,
                task_type="retrieval_document",
            ),
        )
        return result["embedding"]

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        tasks = [self.embed_text(t) for t in texts]
        return list(await asyncio.gather(*tasks))


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider backed by sentence-transformers."""

    def __init__(self, config: EmbeddingConfig):
        self._config = config
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def _lazy_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._config.local_model)
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self._lazy_model
        return self._dimension

    async def embed_text(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(
            None,
            lambda: self._lazy_model.encode(text, convert_to_numpy=True),
        )
        return vec.tolist()

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(
            None,
            lambda: self._lazy_model.encode(
                texts,
                batch_size=self._config.batch_size,
                convert_to_numpy=True,
            ),
        )
        return vecs.tolist()


class EmbeddingProvider:
    """Factory wrapper that delegates to the configured backend provider."""

    _PROVIDER_MAP = {
        "openai": OpenAIEmbeddingProvider,
        "gemini": GeminiEmbeddingProvider,
        "local": LocalEmbeddingProvider,
    }

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self._config = config or EmbeddingConfig()
        self._backend: Optional[BaseEmbeddingProvider] = None

    @property
    def _provider(self) -> BaseEmbeddingProvider:
        if self._backend is None:
            cls = self._PROVIDER_MAP.get(self._config.provider)
            if cls is None:
                raise ValueError(
                    f"Unknown EMBEDDING_PROVIDER: '{self._config.provider}'. "
                    f"Valid options: {list(self._PROVIDER_MAP)}"
                )
            self._backend = cls(self._config)
        return self._backend

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    async def embed_text(self, text: str) -> List[float]:
        return await self._provider.embed_text(text)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return await self._provider.embed_texts(texts)
