"""
Command-line interface for the RAG ingestion pipeline.

Usage:
    python -m rag_ingest ingest <path>         Ingest a file or directory
    python -m rag_ingest stats                 Print database statistics
    python -m rag_ingest delete <doc_id>       Delete a document by ID
"""

import argparse
import asyncio
import sys

from .config import RAGIngestConfig
from .ingestor import RAGIngestor


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_ingest",
        description="RAG ingestion pipeline CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest a Markdown file or directory.")
    ingest_p.add_argument("path", help="Path to a .md file or directory.")
    ingest_p.add_argument(
        "--pattern", default="*.md", help="Glob pattern for directory mode."
    )
    ingest_p.add_argument(
        "--no-recursive", action="store_true", help="Disable recursive directory search."
    )
    ingest_p.add_argument(
        "--doc-id", help="Explicit document ID (file mode only)."
    )

    # stats
    sub.add_parser("stats", help="Print vector and graph store statistics.")

    # delete
    delete_p = sub.add_parser("delete", help="Delete a document from all stores.")
    delete_p.add_argument("doc_id", help="Document ID to delete.")

    return parser


async def _run_ingest(args: argparse.Namespace, ingestor: RAGIngestor) -> None:
    import pathlib
    path = pathlib.Path(args.path)

    if path.is_dir():
        results = await ingestor.ingest_directory(
            directory=str(path),
            pattern=args.pattern,
            recursive=not args.no_recursive,
        )
        ok = sum(1 for r in results if r.success)
        print(f"\nIngested {ok}/{len(results)} file(s) successfully.")
        for r in results:
            status = "OK" if r.success else f"FAILED: {r.error}"
            print(f"  [{status}] {pathlib.Path(r.file_path).name}")
    else:
        result = await ingestor.ingest_markdown(
            str(path), doc_id=getattr(args, "doc_id", None)
        )
        if result.success:
            print(
                f"OK | chunks={result.chunks_count} entities={result.entities_count} "
                f"relationships={result.relationships_count} "
                f"time={result.processing_time:.1f}s"
            )
        else:
            print(f"FAILED: {result.error}", file=sys.stderr)
            sys.exit(1)


async def _run_stats(ingestor: RAGIngestor) -> None:
    stats = await ingestor.get_stats()

    print("--- Vector Store ---")
    for name, info in stats["vector_store"].items():
        if "error" in info:
            print(f"  {name}: ERROR ({info['error']})")
        else:
            print(
                f"  {name}: {info.get('vectors_count', 0)} vectors "
                f"({info.get('status', '?')})"
            )

    print("\n--- Graph Store ---")
    gstats = stats["graph_store"]
    print(f"  Nodes : {gstats.get('nodes_count', 0)}")
    print(f"  Edges : {gstats.get('edges_count', 0)}")
    print(f"  Density: {gstats.get('density', 0.0):.4f}")

    if gstats.get("entity_types"):
        print("\n  Entity types:")
        for t, count in sorted(gstats["entity_types"].items()):
            print(f"    {t}: {count}")


async def _run_delete(doc_id: str, ingestor: RAGIngestor) -> None:
    success = await ingestor.delete_document(doc_id)
    if success:
        print(f"Deleted doc_id={doc_id}")
    else:
        print(f"Failed to delete doc_id={doc_id}", file=sys.stderr)
        sys.exit(1)


async def _main_async() -> None:
    args = _build_parser().parse_args()
    config = RAGIngestConfig()
    ingestor = RAGIngestor(config)

    try:
        if args.command == "ingest":
            await _run_ingest(args, ingestor)
        elif args.command == "stats":
            await _run_stats(ingestor)
        elif args.command == "delete":
            await _run_delete(args.doc_id, ingestor)
    finally:
        ingestor.close()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
