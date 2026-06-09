"""
services/vector_store.py
========================
ChromaDB vector store with progressive embedding fallback.

Embedding strategy (best available wins):
  1. SentenceTransformer('all-MiniLM-L6-v2')  — real embeddings, best quality
  2. FakeEmbedder                              — deterministic fake vectors
     Used when model not downloaded yet.
     Fake embeddings make ChromaDB work end-to-end so you can test
     the full pipeline. Swap to real once download completes.

ChromaDB strategy:
  - Always uses PersistentClient (data survives restarts)
  - Falls back gracefully if chromadb not installed yet
"""

import hashlib
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR = "./data/chroma_db"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_DIM = 384  # Matches all-MiniLM-L6-v2


# ── Embedding Layer ───────────────────────────────────────────────────────────

class FakeEmbedder:
    """
    Deterministic fake embedder.

    Produces consistent 384-dim vectors based on text content.
    NOT semantically meaningful — similar texts get dissimilar vectors.
    Purpose: lets you test ChromaDB storage, retrieval plumbing, and
    the full API pipeline before the real model downloads.

    Why deterministic (not random)?
    - Random vectors change on every run → stored vectors never match queries
    - Deterministic vectors based on text hash → same text always same vector
    - This means "retrieval" still works (returns stored docs), just not
      semantically ranked

    Replace with SentenceTransformerEmbedder once model is downloaded.
    """

    def __init__(self) -> None:
        logger.warning(
            "Using FakeEmbedder — vectors are NOT semantically meaningful. "
            "Download all-MiniLM-L6-v2 and restart to activate real embeddings."
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Produce deterministic pseudo-random vectors from text hashes.
        Each unique text always produces the same vector.
        """
        embeddings = []
        for text in texts:
            # Hash the text to get a deterministic seed
            hash_bytes = hashlib.sha256(text.encode()).digest()

            # Generate 384 floats from the hash (cycling through hash bytes)
            vector = []
            for i in range(EMBEDDING_DIM):
                byte_val = hash_bytes[i % len(hash_bytes)]
                # Normalise to [-1, 1] range
                vector.append((byte_val - 128) / 128.0)

            # L2 normalise so cosine similarity works correctly
            magnitude = math.sqrt(sum(v * v for v in vector))
            if magnitude > 0:
                vector = [v / magnitude for v in vector]

            embeddings.append(vector)

        return embeddings

    def get_sentence_embedding_dimension(self) -> int:
        return EMBEDDING_DIM


class SentenceTransformerEmbedder:
    """
    Real embedder using all-MiniLM-L6-v2.
    Produces semantically meaningful 384-dim vectors.
    """

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model all-MiniLM-L6-v2...")
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ SentenceTransformer loaded")

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [e.tolist() for e in embeddings]

    def get_sentence_embedding_dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


def get_embedder():
    """
    Returns the best available embedder.
    Tries real SentenceTransformer first, falls back to FakeEmbedder.
    """
    try:
        embedder = SentenceTransformerEmbedder()
        return embedder, "real"
    except Exception as e:
        logger.warning(f"SentenceTransformer unavailable ({e}). Using FakeEmbedder.")
        return FakeEmbedder(), "fake"


# ── ChromaDB Client ───────────────────────────────────────────────────────────

def get_chroma_client():
    """
    Returns a ChromaDB client.
    Uses HttpClient if inside Docker, else PersistentClient for local dev.
    """
    try:
        import chromadb
        import os
        from chromadb.config import Settings as ChromaSettings

        chroma_host = os.getenv("CHROMA_HOST")
        chroma_port = os.getenv("CHROMA_PORT", "8000")

        if chroma_host:
            # Running inside Docker Compose - connect to the Chroma service
            logger.info(f"Connecting to remote ChromaDB at {chroma_host}:{chroma_port}")
            client = chromadb.HttpClient(
                host=chroma_host,
                port=int(chroma_port),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        else:
            # Local development fallback
            logger.info("Using local PersistentClient")
            persist_path = Path(CHROMA_PERSIST_DIR)
            persist_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return client
    except ImportError:
        raise ImportError(
            "chromadb is not installed. "
            "Run: pip install chromadb"
        )


def get_collection():
    """Returns the ChromaDB collection, creating it if it doesn't exist."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── Text Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks with metadata.
    Identical logic used in both ingest.py and the API.
    """
    text = " ".join(text.split())  # normalise whitespace
    if not text.strip():
        return []

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # Try to break at sentence/word boundary
        if end < len(text):
            for boundary in [". ", "\n", " "]:
                pos = text.rfind(boundary, start + step, end)
                if pos != -1:
                    end = pos + len(boundary)
                    break

        chunk_text_content = text[start:end].strip()
        if chunk_text_content:
            chunks.append({
                "text": chunk_text_content,
                "source": source,
                "chunk_index": len(chunks),
            })

        start = end - CHUNK_OVERLAP

    # Add total_chunks to all entries now that we know the count
    total = len(chunks)
    for chunk in chunks:
        chunk["total_chunks"] = total

    return chunks


# ── Public API (used by routers) ──────────────────────────────────────────────

def search(
    query: str,
    top_k: int = 3,
    source_filter: Optional[str] = None,
) -> list[dict]:
    """
    Semantic search against ChromaDB.
    Called by routers/search.py and routers/ask.py.

    Returns list of dicts with keys: chunk, score, source, chunk_index
    """
    collection = get_collection()

    if collection.count() == 0:
        logger.warning("Search called on empty collection. Run ingest.py first.")
        return []

    embedder, _ = get_embedder()
    query_embedding = embedder.encode([query])[0]

    where_filter = {"source": source_filter} if source_filter else None
    n_results = min(top_k, collection.count())

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "chunk": doc,
            "score": round(float(1.0 - dist), 4),
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "total_chunks": meta.get("total_chunks", 0),
        })

    return output


def get_collection_stats() -> dict:
    """Returns stats about the current collection. Used by /search/info."""
    collection = get_collection()
    count = collection.count()

    if count == 0:
        return {
            "total_chunks": 0,
            "sources": [],
            "source_count": 0,
            "collection_name": COLLECTION_NAME,
            "persist_dir": CHROMA_PERSIST_DIR,
        }

    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources: dict[str, int] = {}
    for meta in all_meta:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    _, embedder_mode = get_embedder()

    return {
        "total_chunks": count,
        "sources": [{"name": k, "chunks": v} for k, v in sorted(sources.items())],
        "source_count": len(sources),
        "collection_name": COLLECTION_NAME,
        "persist_dir": CHROMA_PERSIST_DIR,
        "embedder_mode": embedder_mode,
    }