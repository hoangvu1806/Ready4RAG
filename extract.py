"""
Script to extract content from PDFs to Markdown using the pdf2md module.

Usage:
    python extract.py <pdf_file> [--provider gemini|openai|groq|ollama]
"""
import sys
from pathlib import Path

# Ensure the project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from pdf2md.cli import main

if __name__ == "__main__":
    main()
