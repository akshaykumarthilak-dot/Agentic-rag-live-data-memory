"""
memory_store.py
----------------
Advanced memory system for Agentic RAG.

Memory types:

    document
        Static knowledge from files in docs/

    episodic
        Long-term memories created from previous conversations

Each memory stores:

    - unique ID
    - text
    - source
    - metadata
    - creation timestamp
    - session ID
    - importance score
    - local semantic embedding

Embeddings are generated locally using sentence-transformers.
"""

import os
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

import config


MemorySource = Literal["document", "episodic"]


@dataclass
class MemoryEntry:

    # ---------------------------------------------------------
    # BASIC MEMORY INFORMATION
    # ---------------------------------------------------------

    id: str

    text: str

    source: MemorySource

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------

    metadata: dict = field(
        default_factory=dict
    )

    # ---------------------------------------------------------
    # ADVANCED MEMORY INFORMATION
    # ---------------------------------------------------------

    created_at: str = ""

    session_id: str = ""

    importance: float = 0.5

    # ---------------------------------------------------------
    # VECTOR EMBEDDING
    # ---------------------------------------------------------

    embedding: np.ndarray = None

    # ---------------------------------------------------------
    # BACKWARD COMPATIBILITY
    # ---------------------------------------------------------

    def __setstate__(self, state):

        self.__dict__.update(state)

        # Older memories may not have these fields.
        if not hasattr(self, "created_at"):
            self.created_at = ""

        if not hasattr(self, "session_id"):
            self.session_id = ""

        if not hasattr(self, "importance"):
            self.importance = 0.5

        if not hasattr(self, "metadata"):
            self.metadata = {}


class VectorMemoryStore:

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME
    ):

        # -----------------------------------------------------
        # LOCAL EMBEDDING MODEL
        # -----------------------------------------------------

        self._model = SentenceTransformer(
            model_name
        )

        # -----------------------------------------------------
        # MEMORY ENTRIES
        # -----------------------------------------------------

        self.entries: List[MemoryEntry] = []

        # -----------------------------------------------------
        # EMBEDDING MATRIX
        # -----------------------------------------------------

        self._embeddings_matrix: Optional[
            np.ndarray
        ] = None

        # -----------------------------------------------------
        # LOAD EXISTING MEMORY
        # -----------------------------------------------------

        self._load()

    # =========================================================
    # EMBEDDING HELPERS
    # =========================================================

    def _embed(
        self,
        texts: List[str]
    ) -> np.ndarray:

        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return vectors

    def _rebuild_matrix(self):

        if not self.entries:

            self._embeddings_matrix = None

            return

        valid_entries = [
            entry
            for entry in self.entries
            if entry.embedding is not None
        ]

        if not valid_entries:

            self._embeddings_matrix = None

            return

        self._embeddings_matrix = np.stack(
            [
                entry.embedding
                for entry in valid_entries
            ]
        )

    # =========================================================
    # ADD SINGLE MEMORY
    # =========================================================

    def add(
        self,
        text: str,
        source: MemorySource,
        metadata: dict = None,
        session_id: str = "",
        importance: float = 0.5
    ) -> str:

        entry_id = str(
            uuid.uuid4()
        )

        embedding = self._embed(
            [text]
        )[0]

        created_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        # Keep importance between 0 and 1
        importance = max(
            0.0,
            min(
                1.0,
                float(importance)
            )
        )

        entry = MemoryEntry(

            id=entry_id,

            text=text,

            source=source,

            metadata=metadata or {},

            created_at=created_at,

            session_id=session_id,

            importance=importance,

            embedding=embedding
        )

        self.entries.append(
            entry
        )

        self._rebuild_matrix()

        self._save()

        return entry_id

    # =========================================================
    # ADD MULTIPLE MEMORIES
    # =========================================================

    def add_batch(
        self,
        texts: List[str],
        source: MemorySource,
        metadata_list: List[dict] = None,
        session_id: str = "",
        importance: float = 0.5
    ):

        if not texts:

            return

        metadata_list = (
            metadata_list
            or [{} for _ in texts]
        )

        embeddings = self._embed(
            texts
        )

        created_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        importance = max(
            0.0,
            min(
                1.0,
                float(importance)
            )
        )

        for text, meta, emb in zip(
            texts,
            metadata_list,
            embeddings
        ):

            self.entries.append(

                MemoryEntry(

                    id=str(
                        uuid.uuid4()
                    ),

                    text=text,

                    source=source,

                    metadata=meta,

                    created_at=created_at,

                    session_id=session_id,

                    importance=importance,

                    embedding=emb
                )
            )

        self._rebuild_matrix()

        self._save()

    # =========================================================
    # BASIC SEMANTIC SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
        source_filter: Literal[
            "document",
            "episodic",
            "any"
        ] = "any",
        min_score: float = -1.0
    ) -> List[MemoryEntry]:

        results = self.search_with_scores(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
            min_score=min_score
        )

        return [
            entry
            for entry, score in results
        ]

    # =========================================================
    # SEARCH WITH SIMILARITY SCORES
    # =========================================================

    def search_with_scores(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVAL,
        source_filter: Literal[
            "document",
            "episodic",
            "any"
        ] = "any",
        min_score: float = -1.0
    ):

        if (
            self._embeddings_matrix is None
            or len(self.entries) == 0
        ):

            return []

        query_vec = self._embed(
            [query]
        )[0]

        candidates = []

        for index, entry in enumerate(
            self.entries
        ):

            if (
                source_filter != "any"
                and entry.source != source_filter
            ):

                continue

            candidates.append(
                (index, entry)
            )

        if not candidates:

            return []

        indexes = [
            index
            for index, entry in candidates
        ]

        sub_matrix = (
            self._embeddings_matrix[indexes]
        )

        # Cosine similarity because vectors
        # are normalized.
        scores = (
            sub_matrix @ query_vec
        )

        ranked_indexes = np.argsort(
            -scores
        )

        results = []

        for ranked_index in ranked_indexes:

            original_index = indexes[
                ranked_index
            ]

            entry = self.entries[
                original_index
            ]

            score = float(
                scores[ranked_index]
            )

            if score < min_score:

                continue

            results.append(
                (
                    entry,
                    score
                )
            )

            if len(results) >= top_k:

                break

        return results

    # =========================================================
    # SEARCH BY SESSION
    # =========================================================

    def search_session(
        self,
        session_id: str,
        top_k: int = config.TOP_K_RETRIEVAL
    ) -> List[MemoryEntry]:

        session_entries = [

            entry

            for entry in self.entries

            if entry.session_id == session_id

        ]

        # Most recent first
        session_entries.sort(
            key=lambda entry: entry.created_at,
            reverse=True
        )

        return session_entries[
            :top_k
        ]

    # =========================================================
    # COUNT MEMORIES
    # =========================================================

    def count(
        self,
        source: Literal[
            "document",
            "episodic",
            "any"
        ] = "any"
    ) -> int:

        if source == "any":

            return len(
                self.entries
            )

        return sum(

            1

            for entry in self.entries

            if entry.source == source

        )

    # =========================================================
    # MEMORY STATISTICS
    # =========================================================

    def statistics(self) -> dict:

        document_count = self.count(
            "document"
        )

        episodic_count = self.count(
            "episodic"
        )

        total_count = self.count(
            "any"
        )

        return {

            "total": total_count,

            "documents": document_count,

            "episodic": episodic_count,

        }

    # =========================================================
    # PERSISTENCE
    # =========================================================

    def _save(self):

        with open(
            config.VECTOR_STORE_PATH,
            "wb"
        ) as f:

            pickle.dump(
                self.entries,
                f
            )

    def _load(self):

        if not os.path.exists(
            config.VECTOR_STORE_PATH
        ):

            return

        try:

            with open(
                config.VECTOR_STORE_PATH,
                "rb"
            ) as f:

                self.entries = pickle.load(
                    f
                )

            # Make sure old entries have
            # the new attributes.
            for entry in self.entries:

                if not hasattr(
                    entry,
                    "created_at"
                ):

                    entry.created_at = ""

                if not hasattr(
                    entry,
                    "session_id"
                ):

                    entry.session_id = ""

                if not hasattr(
                    entry,
                    "importance"
                ):

                    entry.importance = 0.5

                if not hasattr(
                    entry,
                    "metadata"
                ):

                    entry.metadata = {}

            self._rebuild_matrix()

        except (
            pickle.PickleError,
            EOFError,
            AttributeError,
            ValueError
        ):

            # If the old vector store is corrupted,
            # start with an empty store.
            self.entries = []

            self._embeddings_matrix = None