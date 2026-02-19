"""
LLM provider abstraction for PDF-to-Markdown conversion.

Each provider implements the same interface: receive a PNG image as bytes and
a text prompt, return the extracted Markdown string.
"""

from abc import ABC, abstractmethod
from typing import Optional

import requests


class LLMProvider(ABC):
    """Abstract base for vision LLM providers used in PDF-to-Markdown conversion."""

    @abstractmethod
    def extract_text(self, image_data: bytes, prompt: str) -> str:
        """Return Markdown text extracted from the given page image."""


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible vision provider (supports custom base_url for proxies)."""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self._api_key = api_key
        self._model = model
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    def extract_text(self, image_data: bytes, prompt: str) -> str:
        import base64

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/png;base64,"
                                    + base64.b64encode(image_data).decode()
                                )
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        }

        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class GeminiProvider(LLMProvider):
    """Google Gemini vision provider via the REST generateContent API."""

    _API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def extract_text(self, image_data: bytes, prompt: str) -> str:
        import base64

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": base64.b64encode(image_data).decode(),
                            }
                        },
                    ]
                }
            ]
        }

        response = requests.post(
            f"{self._API_BASE}/{self._model}:generateContent?key={self._api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )

        if not response.ok:
            raise RuntimeError(
                f"Gemini API error [{self._model}] "
                f"{response.status_code}: {response.text}"
            )

        return response.json()["candidates"][0]["content"]["parts"][0]["text"]


class GroqProvider(LLMProvider):
    """Groq vision provider (OpenAI-compatible chat completions endpoint)."""

    _API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        self._api_key = api_key
        self._model = model

    def extract_text(self, image_data: bytes, prompt: str) -> str:
        import base64

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/png;base64,"
                                    + base64.b64encode(image_data).decode()
                                )
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        }

        response = requests.post(
            self._API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )

        if not response.ok:
            raise RuntimeError(
                f"Groq API error [{self._model}] "
                f"{response.status_code}: {response.text}"
            )

        return response.json()["choices"][0]["message"]["content"]


class OllamaProvider(LLMProvider):
    """Ollama local vision provider (uses the /api/generate endpoint)."""

    def __init__(self, host: str, model: str):
        self._host = host.rstrip("/")
        self._model = model

    def extract_text(self, image_data: bytes, prompt: str) -> str:
        import base64

        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [base64.b64encode(image_data).decode()],
            "stream": False,
        }

        response = requests.post(
            f"{self._host}/api/generate",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["response"]


def create_provider(config) -> LLMProvider:
    """Factory: instantiate the LLM provider described in config."""
    if config.llm_provider == "openai":
        return OpenAIProvider(
            api_key=config.openai_api_key,
            model=config.openai_model,
            base_url=config.openai_base_url,
        )
    if config.llm_provider == "gemini":
        return GeminiProvider(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
        )
    if config.llm_provider == "groq":
        return GroqProvider(
            api_key=config.groq_api_key,
            model=config.groq_model,
        )
    if config.llm_provider == "ollama":
        return OllamaProvider(
            host=config.ollama_host,
            model=config.ollama_model,
        )
    raise ValueError(f"Unsupported PDF2MD_LLM_PROVIDER: '{config.llm_provider}'")
