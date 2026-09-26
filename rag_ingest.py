"""
rag_ingest.py
-------------
Run this once (and again whenever you add new files to docs/) to build
your static knowledge base. It reads every .txt file in docs/, splits it
into overlapping chunks, and stores the chunks as "document" memories.

Usage:
    python rag_ingest.py
"""

import os
import config
from memory_store import VectorMemoryStore


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list:
    """
    Simple sliding-window chunker. Real production systems often chunk by
    sentence/paragraph boundaries instead of raw character counts — this
    is deliberately simple so it's easy to explain in an interview.
    """
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step forward, keeping some overlap for context continuity
    return [c for c in chunks if c.strip()]


def ingest_docs():
    store = VectorMemoryStore()
    docs_loaded = 0
    chunks_loaded = 0

    for filename in os.listdir(config.DOCS_DIR):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(config.DOCS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        metadata_list = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
        store.add_batch(chunks, source="document", metadata_list=metadata_list)

        docs_loaded += 1
        chunks_loaded += len(chunks)
        print(f"  Ingested {filename}: {len(chunks)} chunks")

    print(f"\nDone. {docs_loaded} file(s) -> {chunks_loaded} chunk(s) added to the vector store.")
    print(f"Total memory entries now: {store.count()} "
          f"(documents: {store.count('document')}, episodic: {store.count('episodic')})")


if __name__ == "__main__":
    ingest_docs()
