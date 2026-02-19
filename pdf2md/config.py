"""
Configuration for the PDF-to-Markdown conversion module.

All values are read from environment variables (or a .env file at the project
root). Defaults reflect recommended, cost-effective model choices.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default, cast: type = str):
    raw = os.getenv(key)
    if raw is None:
        return default
    if cast is bool:
        return raw.lower() in ("true", "1", "yes")
    if cast is int:
        return int(raw)
    return raw


@dataclass
class PDF2MDConfig:
    """Configuration for PDF-to-Markdown conversion."""

    # Output directory for generated Markdown files
    output_dir: str = field(
        default_factory=lambda: _env("PDF2MD_OUTPUT_DIR", "./output/markdown")
    )

    # LLM provider used to OCR each PDF page image
    # Accepted values: openai | gemini | groq | ollama
    llm_provider: str = field(
        default_factory=lambda: _env("PDF2MD_LLM_PROVIDER", "gemini")
    )

    # --- OpenAI ---
    openai_api_key: Optional[str] = field(
        default_factory=lambda: _env("OPENAI_API_KEY", None)
    )
    openai_base_url: str = field(
        default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    # gpt-4o-mini supports vision at low cost; upgrade to gpt-4o for higher accuracy.
    openai_model: str = field(
        default_factory=lambda: _env("OPENAI_VISION_MODEL", "gpt-4o-mini")
    )

    # --- Google Gemini ---
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: _env("GEMINI_API_KEY", None)
    )
    # gemini-2.0-flash is the current stable multimodal release (GA Feb 2025).
    gemini_model: str = field(
        default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.0-flash")
    )

    # --- Groq ---
    groq_api_key: Optional[str] = field(
        default_factory=lambda: _env("GROQ_API_KEY", None)
    )
    # llama-4-scout-17b-16e-instruct is the current vision-capable model on Groq.
    groq_model: str = field(
        default_factory=lambda: _env(
            "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
        )
    )

    # --- Ollama (local) ---
    ollama_host: str = field(
        default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: _env("OLLAMA_VISION_MODEL", "llava")
    )

    # Maximum dimension of the page image sent to the LLM (width and height).
    # Larger values improve accuracy at the cost of higher API token usage.
    image_size: int = field(
        default_factory=lambda: _env("PDF2MD_IMAGE_SIZE", 1024, int)
    )

    # When True, LaTeX syntax is used for all mathematical expressions.
    preserve_formulas: bool = field(
        default_factory=lambda: _env("PDF2MD_PRESERVE_FORMULAS", True, bool)
    )

    # When True, embedded images are extracted and saved alongside the Markdown.
    extract_images: bool = field(
        default_factory=lambda: _env("PDF2MD_EXTRACT_IMAGES", False, bool)
    )

    # Number of concurrent page-processing threads.
    max_workers: int = field(
        default_factory=lambda: _env("PDF2MD_MAX_WORKERS", 4, int)
    )

    def validate(self) -> None:
        """Raise ValueError if the selected provider is mis-configured."""
        valid_providers = ("openai", "gemini", "groq", "ollama")
        if self.llm_provider not in valid_providers:
            raise ValueError(
                f"Invalid PDF2MD_LLM_PROVIDER '{self.llm_provider}'. "
                f"Must be one of: {valid_providers}"
            )

        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when PDF2MD_LLM_PROVIDER=openai")
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when PDF2MD_LLM_PROVIDER=gemini")
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when PDF2MD_LLM_PROVIDER=groq")
