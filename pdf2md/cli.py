"""
Command-line interface for PDF-to-Markdown conversion.

Usage:
    python -m pdf2md <pdf_path> [options]

Examples:
    python -m pdf2md report.pdf
    python -m pdf2md report.pdf --provider gemini --output report.md
    python -m pdf2md report.pdf --provider openai --model gpt-4o --workers 8
"""

import argparse

from .config import PDF2MDConfig
from .converter import PDFToMarkdownConverter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md",
        description="Convert a PDF document to Markdown using a vision LLM.",
    )

    parser.add_argument("pdf_path", help="Path to the source PDF file.")

    parser.add_argument(
        "-o", "--output",
        metavar="PATH",
        help="Destination Markdown file path. Default: auto-generated in output_dir.",
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["openai", "gemini", "groq", "ollama"],
        help="LLM provider to use. Overrides PDF2MD_LLM_PROVIDER from .env.",
    )
    parser.add_argument(
        "-m", "--model",
        help="Model name for the selected provider.",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        help="Number of concurrent page-processing threads. Default: 4.",
    )
    parser.add_argument(
        "--no-formulas",
        action="store_true",
        help="Disable LaTeX formula preservation in the extraction prompt.",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = PDF2MDConfig()

    if args.provider:
        config.llm_provider = args.provider

    if args.model:
        provider = config.llm_provider
        model_attr = {
            "openai": "openai_model",
            "gemini": "gemini_model",
            "groq": "groq_model",
            "ollama": "ollama_model",
        }.get(provider)
        if model_attr:
            setattr(config, model_attr, args.model)

    if args.workers:
        config.max_workers = args.workers

    if args.no_formulas:
        config.preserve_formulas = False

    converter = PDFToMarkdownConverter(config)
    output_path = converter.convert_file(args.pdf_path, args.output)
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
