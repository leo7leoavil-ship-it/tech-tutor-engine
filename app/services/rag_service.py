"""ChromaDB singleton wrapper over the pre-built `tech_tutor_corpus` collection."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.schemas.response_schemas import RetrievedChunk

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)


class RAGService:
    """Persistent ChromaDB client bound to the Data Track corpus.

    The collection is **never** created or re-embedded here. The Data Track
    already persisted `tech_tutor_corpus` under `chroma_persist_dir`.
    """

    _instance: RAGService | None = None
    _lock = threading.Lock()

    def __new__(cls, *args: object, **kwargs: object) -> RAGService:
        """Process-wide singleton so embedding weights load once."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, settings: Settings) -> None:
        if getattr(self, "_initialized", False):
            return
        self._settings = settings
        self._collection: Collection | None = None
        self._initialized = True

    def connect(self) -> None:
        """Open the persistent client and bind the existing collection."""
        persist_path = Path(self._settings.chroma_persist_dir).resolve()
        if not persist_path.exists():
            raise FileNotFoundError(
                f"ChromaDB persist directory not found: {persist_path}. "
                "Do not rebuild the vector store; copy the Data Track artifact."
            )

        self._collection = open_persistent_collection(
            persist_path=persist_path,
            collection_name=self._settings.chroma_collection_name,
            embedding_model_name=self._settings.embedding_model_name,
        )
        logger.info(
            "Bound ChromaDB collection=%s path=%s count=%s",
            self._settings.chroma_collection_name,
            persist_path,
            self._collection.count(),
        )

    @property
    def is_ready(self) -> bool:
        """Whether the existing collection handle is open."""
        return self._collection is not None

    def collection_count(self) -> int:
        """Number of persisted embeddings (0 if not connected)."""
        if self._collection is None:
            return 0
        return int(self._collection.count())

    async def query_context(self, query_text: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Retrieve top-k cosine neighbours for an engineer utterance.

        Args:
            query_text: Natural-language query (typically the ASR transcript).
            top_k: Neighbour count; cosine similarity is Chroma's default space
                when the collection was built with MiniLM embeddings.

        Returns:
            Ranked chunks including `source` metadata for repo traceability.
        """
        if self._collection is None:
            raise RuntimeError("RAGService.connect() must be called during startup.")
        if not query_text.strip():
            return []
        return await asyncio.to_thread(self._query_sync, query_text, top_k)

    def _query_sync(self, query_text: str, top_k: int) -> list[RetrievedChunk]:
        """Blocking Chroma query executed on a worker thread."""
        assert self._collection is not None
        result = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            meta = dict(metadata or {})
            source = _extract_source(meta)
            chunks.append(
                RetrievedChunk(
                    document=document or "",
                    metadata={k: _coerce_meta(v) for k, v in meta.items()},
                    distance=float(distance) if distance is not None else None,
                    source=source,
                )
            )
        return chunks


def _extract_source(metadata: dict[str, object]) -> str | None:
    """Prefer explicit source keys used by the Data Track ingest pipeline."""
    for key in ("source", "repo", "repository", "path", "file", "origin"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _coerce_meta(value: object) -> str | int | float | None:
    """Narrow Chroma metadata values to the response schema union."""
    if value is None or isinstance(value, (str, int, float)):
        return value
    return str(value)


def open_persistent_collection(
    persist_path: Path,
    collection_name: str,
    embedding_model_name: str,
):
    """Bind the existing Chroma collection with MiniLM embeddings.

    Isolated for tests so Chroma/SentenceTransformer are not imported in unit tests.
    """
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
    client = chromadb.PersistentClient(path=str(persist_path))
    return client.get_collection(name=collection_name, embedding_function=embedding_fn)


def reset_rag_singleton() -> None:
    """Test helper: drop the process singleton between fixtures."""
    RAGService._instance = None
