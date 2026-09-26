"""
memory_store.py
----------------
The "memory" half of Agentic RAG.

Two kinds of memory live in the SAME vector index, distinguished by a
`source` tag:

  - "document"    -> chunks from files in docs/  (classic RAG knowledge base)
  - "episodic"    -> summaries of past conversations (long-term agent memory)

Why one store instead of two?  Because at query time the agent doesn't
care WHERE a fact came from — it just wants "what's relevant to this
question" — and a single similarity search over everything is simpler
and often more accurate than merging two separate result sets.

Embeddings are computed locally with sentence-transformers, so there's no
API cost or network dependency for the retrieval step — only the final
reasoning step calls the LLM.
"""

import os
import pickle
import uuid
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
from sentence_transformers import SentenceTransformer

import config


@dataclass
class MemoryEntry:
    id: str
    text: str
    source: Literal["document", "episodic"]
    metadata: dict = field(default_factory=dict)
    embedding: np.ndarray = None  # set after embedding


class VectorMemoryStore:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self._model = SentenceTransformer(model_name)
        self.entries: List[MemoryEntry] = []
        self._embeddings_matrix: np.ndarray = None  # (N, dim), rebuilt lazily
        self._load()

    # ---------- embedding helpers ----------

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors

    def _rebuild_matrix(self):
        if not self.entries:
            self._embeddings_matrix = None
            return
        self._embeddings_matrix = np.stack([e.embedding for e in self.entries])

    # ---------- public API ----------

    def add(self, text: str, source: Literal["document", "episodic"], metadata: dict = None) -> str:
        entry_id = str(uuid.uuid4())
        embedding = self._embed([text])[0]
        entry = MemoryEntry(id=entry_id, text=text, source=source,
                             metadata=metadata or {}, embedding=embedding)
        self.entries.append(entry)
        self._rebuild_matrix()
        self._save()
        return entry_id

    def add_batch(self, texts: List[str], source: Literal["document", "episodic"],
                  metadata_list: List[dict] = None):
        if not texts:
            return
        metadata_list = metadata_list or [{} for _ in texts]
        embeddings = self._embed(texts)
        for text, meta, emb in zip(texts, metadata_list, embeddings):
            self.entries.append(MemoryEntry(id=str(uuid.uuid4()), text=text,
                                             source=source, metadata=meta, embedding=emb))
        self._rebuild_matrix()
        self._save()

    def search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL,
               source_filter: Literal["document", "episodic", "any"] = "any") -> List[MemoryEntry]:
        if self._embeddings_matrix is None or len(self.entries) == 0:
            return []

        query_vec = self._embed([query])[0]

        candidates = [(i, e) for i, e in enumerate(self.entries)
                      if source_filter == "any" or e.source == source_filter]
        if not candidates:
            return []

        idxs = [i for i, _ in candidates]
        sub_matrix = self._embeddings_matrix[idxs]

        # cosine similarity (vectors are already normalized, so this is a dot product)
        scores = sub_matrix @ query_vec
        ranked = np.argsort(-scores)[:top_k]

        return [candidates[i][1] for i in ranked]

    def count(self, source: Literal["document", "episodic", "any"] = "any") -> int:
        if source == "any":
            return len(self.entries)
        return sum(1 for e in self.entries if e.source == source)

    # ---------- persistence ----------

    def _save(self):
        with open(config.VECTOR_STORE_PATH, "wb") as f:
            pickle.dump(self.entries, f)

    def _load(self):
        if os.path.exists(config.VECTOR_STORE_PATH):
            with open(config.VECTOR_STORE_PATH, "rb") as f:
                self.entries = pickle.load(f)
            self._rebuild_matrix()
