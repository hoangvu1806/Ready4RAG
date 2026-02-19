"""
Script to ingest extracted Markdown content into the RAG system (Vector + Graph DB).

Usage:
    python ingest.py ingest <file_or_dir>
    python ingest.py stats
    python ingest.py delete <doc_id>
"""
import sys
import asyncio
from pathlib import Path

# Ensure the project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from rag_ingest.cli import main

if __name__ == "__main__":
    main()
