"""
ingest.py
=========
Reads .txt files from data/, chunks them, embeds, and stores in ChromaDB.

Works in two embedding modes:
  - real: SentenceTransformer all-MiniLM-L6-v2 (semantic embeddings)
  - fake: FakeEmbedder (deterministic dummy vectors, no model needed)

ChromaDB is always real (requires: pip install chromadb).

Usage:
  python ingest.py              # ingest all .txt in data/
  python ingest.py --overwrite  # re-ingest, replacing existing chunks
  python ingest.py --stats      # show collection info and exit
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing chunks")
    parser.add_argument("--stats", action="store_true", help="Show stats and exit")
    parser.add_argument("--dir", type=Path, default=Path("data"), help="Source directory")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  FastAPI RAG — Ingestion Script")
    print("=" * 55)

    # Import after argparse so --help works even without packages
    try:
        from services.vector_store import (
            get_collection,
            get_collection_stats,
            get_embedder,
            chunk_text,
        )
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("   Run: pip install chromadb")
        sys.exit(1)

    # Check embedder mode
    embedder, embedder_mode = get_embedder()
    print(f"\n  Embedder : {embedder_mode.upper()} "
          f"({'all-MiniLM-L6-v2' if embedder_mode == 'real' else 'FakeEmbedder — dummy vectors'})")

    if embedder_mode == "fake":
        print(
            "\n  ⚠️  WARNING: Using fake embeddings.\n"
            "  Retrieval will work but results won't be semantically ranked.\n"
            "  Once all-MiniLM-L6-v2 downloads, delete data/chroma_db/ and re-run.\n"
        )

    # Stats-only mode
    if args.stats:
        stats = get_collection_stats()
        print(f"\n  Collection : {stats['collection_name']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Sources: {stats['source_count']}")
        for src in stats.get("sources", []):
            print(f"    📄 {src['name']}: {src['chunks']} chunks")
        print()
        return

    # Find files
    data_dir = args.dir
    if not data_dir.exists():
        print(f"\n❌ Directory not found: {data_dir}")
        print("   Create it and add .txt files: mkdir data")
        sys.exit(1)

    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"\n⚠️  No .txt files found in {data_dir}/")
        print("   Add some .txt files and re-run.")
        sys.exit(0)

    print(f"\n  Found {len(txt_files)} .txt file(s) in {data_dir}/\n")

    collection = get_collection()
    total_start = time.perf_counter()
    ingested = skipped = failed = 0

    for file_path in txt_files:
        print(f"  Processing: {file_path.name} ...", end=" ", flush=True)

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if len(text) < 50:
                print("⏭  SKIPPED (too short)")
                skipped += 1
                continue

            # Check if already exists
            existing = collection.get(
                where={"source": file_path.name},
                include=[],
            )
            if existing["ids"] and not args.overwrite:
                print(f"⏭  SKIPPED (already ingested, use --overwrite)")
                skipped += 1
                continue

            if existing["ids"] and args.overwrite:
                collection.delete(where={"source": file_path.name})

            # Chunk
            chunks = chunk_text(text, source=file_path.name)
            if not chunks:
                print("⏭  SKIPPED (no chunks generated)")
                skipped += 1
                continue

            # Embed
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embedder.encode(chunk_texts)

            # Build IDs and metadata
            import hashlib
            ids = [
                hashlib.md5(f"{c['source']}_{c['chunk_index']}".encode()).hexdigest()
                for c in chunks
            ]
            metadatas = [
                {
                    "source": c["source"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                }
                for c in chunks
            ]

            # Store
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas,
            )

            print(f"✅ {len(chunks)} chunks")
            ingested += 1

        except Exception as e:
            print(f"❌ FAILED: {e}")
            logger.error(f"Failed to ingest {file_path.name}", exc_info=True)
            failed += 1

    elapsed = time.perf_counter() - total_start

    print(f"\n{'=' * 55}")
    print(f"  ✅ Ingested : {ingested} files")
    print(f"  ⏭  Skipped  : {skipped} files")
    print(f"  ❌ Failed   : {failed} files")
    print(f"  ⏱  Time     : {elapsed:.1f}s")
    print(f"  📦 Total chunks in DB: {collection.count()}")
    print(f"{'=' * 55}\n")

    if embedder_mode == "fake":
        print(
            "  Reminder: delete data/chroma_db/ and re-run ingest.py\n"
            "  once the real model finishes downloading.\n"
        )


if __name__ == "__main__":
    main()