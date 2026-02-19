"""
Test script for RAG Ingestion Pipeline.

Usage:
    python -m services.ingestion.rag_ingest.test_ingest
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.ingestion.rag_ingest import RAGIngestor, RAGIngestConfig


async def test_single_file():
    """Test ingesting a single markdown file."""
    print("=" * 60)
    print("TEST: Single File Ingestion")
    print("=" * 60)
    
    # Find a test file
    test_files = list(Path("pdf2md_output").rglob("*.md"))
    
    if not test_files:
        print("No markdown files found in pdf2md_output/")
        print("Please run pdf2md first to generate some markdown files.")
        return
    
    test_file = test_files[0]
    print(f"Using test file: {test_file}")
    
    # Create config
    config = RAGIngestConfig()
    
    # Initialize ingestor
    ingestor = RAGIngestor(config)
    
    try:
        # Ingest the file
        result = await ingestor.ingest_markdown(str(test_file))
        
        print("\n" + "-" * 40)
        print("RESULT:")
        print(f"  Success: {result.success}")
        print(f"  Doc ID: {result.doc_id}")
        print(f"  Chunks: {result.chunks_count}")
        print(f"  Entities: {result.entities_count}")
        print(f"  Relationships: {result.relationships_count}")
        print(f"  Time: {result.processing_time:.2f}s")
        
        if result.error:
            print(f"  Error: {result.error}")
        
        # Get stats
        print("\n" + "-" * 40)
        print("DATABASE STATS:")
        stats = await ingestor.get_stats()
        
        print("\nVector Store:")
        for name, collection_stats in stats["vector_store"].items():
            print(f"  {name}: {collection_stats.get('vectors_count', 0)} vectors")
        
        print("\nGraph Store:")
        print(f"  Nodes: {stats['graph_store']['nodes_count']}")
        print(f"  Edges: {stats['graph_store']['edges_count']}")
        
    finally:
        ingestor.close()


async def test_directory():
    """Test ingesting a directory of markdown files."""
    print("\n" + "=" * 60)
    print("TEST: Directory Ingestion")
    print("=" * 60)
    
    config = RAGIngestConfig()
    ingestor = RAGIngestor(config)
    
    try:
        results = await ingestor.ingest_directory(
            "pdf2md_output",
            pattern="*.md",
            recursive=True
        )
        
        print("\n" + "-" * 40)
        print("RESULTS:")
        
        for result in results:
            status = "OK" if result.success else "FAILED"
            print(f"  [{status}] {Path(result.file_path).name}: {result.chunks_count} chunks, {result.entities_count} entities")
        
    finally:
        ingestor.close()


async def test_search():
    """Test search functionality."""
    print("\n" + "=" * 60)
    print("TEST: Vector Search")
    print("=" * 60)
    
    config = RAGIngestConfig()
    ingestor = RAGIngestor(config)
    
    try:
        await ingestor.initialize()
        
        # Search for chunks
        query = "tim mach"  # Heart-related query
        print(f"\nSearching for: '{query}'")
        
        query_embedding = await ingestor.embedding_provider.embed_text(query)
        
        results = await ingestor.vector_store.search_chunks(
            query_embedding=query_embedding,
            top_k=3
        )
        
        print(f"\nTop {len(results)} results:")
        for i, result in enumerate(results, 1):
            content = result.payload.get("content", "")[:200]
            print(f"\n{i}. Score: {result.score:.4f}")
            print(f"   Content: {content}...")
        
    finally:
        ingestor.close()


async def main():
    """Run all tests."""
    print("RAG Ingestion Pipeline Test Suite")
    print("=" * 60)
    
    # Check if pdf2md_output exists
    if not Path("pdf2md_output").exists():
        print("Warning: pdf2md_output/ directory not found.")
        print("Creating sample markdown for testing...")
        
        # Create sample markdown
        sample_dir = Path("pdf2md_output/test")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        sample_content = """# Sample Medical Document

## Introduction

This is a sample document about cardiovascular health and heart disease prevention.

---

## Cardiovascular System

The cardiovascular system consists of the heart, blood vessels, and blood. The heart pumps blood throughout the body, delivering oxygen and nutrients to cells.

### Heart Anatomy

The heart has four chambers:
- Right atrium
- Right ventricle
- Left atrium
- Left ventricle

---

## Risk Factors

Common risk factors for heart disease include:

1. High blood pressure (hypertension)
2. High cholesterol
3. Smoking
4. Diabetes
5. Obesity
6. Physical inactivity

### Prevention

Regular exercise and a healthy diet can significantly reduce the risk of cardiovascular disease.

---

## Conclusion

Understanding cardiovascular health is essential for disease prevention and maintaining overall well-being.
"""
        
        sample_file = sample_dir / "sample_medical.md"
        sample_file.write_text(sample_content, encoding="utf-8")
        print(f"Created sample file: {sample_file}")
    
    # Run tests
    await test_single_file()
    # await test_directory()  # Uncomment to test directory ingestion
    await test_search()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
