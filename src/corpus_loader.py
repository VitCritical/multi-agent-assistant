"""
corpus_loader.py — Load corpus.jsonl into ChromaDB using local sentence-transformers embeddings.
Run once; subsequent starts skip re-embedding if collection already populated.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.config import (
    CORPUS_PATH, CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL, TOP_K_CHUNKS
)


# ── Singleton embedding model ─────────────────────────────────────
_embed_model: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"[corpus_loader] Loading embedding model: {EMBEDDING_MODEL}")
        try:
            # Prefer local cache without network call if already downloaded
            _embed_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except Exception:
            _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


# ── Singleton ChromaDB client ─────────────────────────────────────
_chroma_client: Optional[Any] = None

def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    return get_chroma_client().get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def load_corpus() -> List[Dict]:
    """Parse corpus.jsonl — each line is one chunk."""
    corpus_path = Path(CORPUS_PATH)
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus.jsonl not found at {corpus_path}. "
                                "Make sure it is in the project root.")
    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def init_vector_store(force_reload: bool = False) -> None:
    """
    Embed corpus and persist to ChromaDB.
    Skips if collection already has documents (unless force_reload=True).
    """
    collection = get_collection()
    if collection.count() > 0 and not force_reload:
        print(f"[corpus_loader] ChromaDB already has {collection.count()} chunks — skipping re-embed.")
        return

    chunks = load_corpus()
    print(f"[corpus_loader] Embedding {len(chunks)} chunks from corpus.jsonl …")

    model = get_embed_model()
    texts, ids, metadatas = [], [], []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        # build rich text for embedding: title + content
        title = chunk.get("title", "")
        content = chunk.get("content", chunk.get("text", ""))
        text_to_embed = f"{title}\n\n{content}".strip()

        texts.append(text_to_embed)
        ids.append(chunk_id)
        metadatas.append({
            "chunk_id":   chunk_id,
            "doc_id":     chunk.get("doc_id", ""),
            "title":      title,
            "category":   chunk.get("category", ""),
            "published":  str(chunk.get("published", "")),
            "version":    str(chunk.get("version", "")),
            "source_url": chunk.get("source_url", ""),
        })

    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # upsert in batches
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )

    print(f"[corpus_loader] ✓ {collection.count()} chunks stored in ChromaDB.")


def semantic_search(query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict]:
    """
    Embed query and retrieve top_k chunks.
    Returns list of dicts with chunk content + metadata.
    """
    model = get_embed_model()
    query_embedding = model.encode([query]).tolist()[0]
    collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "chunk_id":   meta["chunk_id"],
            "doc_id":     meta["doc_id"],
            "title":      meta["title"],
            "category":   meta["category"],
            "published":  meta["published"],
            "version":    meta["version"],
            "source_url": meta["source_url"],
            "content":    doc,
            "score":      round(1 - dist, 4),   # cosine similarity
        })

    return chunks
