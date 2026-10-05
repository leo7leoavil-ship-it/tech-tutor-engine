"""Unit tests for the ChromaDB RAG singleton."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings
from app.services.rag_service import RAGService, reset_rag_singleton


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    reset_rag_singleton()
    yield
    reset_rag_singleton()


def test_connect_fails_when_persist_dir_missing(tmp_path: Path) -> None:
    settings = Settings(chroma_persist_dir=str(tmp_path / "missing-db"))
    service = RAGService(settings)
    with pytest.raises(FileNotFoundError, match="ChromaDB persist directory"):
        service.connect()


def test_query_context_returns_source_traceability(tmp_path: Path) -> None:
    persist = tmp_path / "technical_corpus_db"
    persist.mkdir()
    settings = Settings(
        chroma_persist_dir=str(persist),
        chroma_collection_name="tech_tutor_corpus",
        embedding_model_name="all-MiniLM-L6-v2",
    )

    fake_collection = MagicMock()
    fake_collection.count.return_value = 42
    fake_collection.query.return_value = {
        "documents": [["maxSurge controls extra pods"]],
        "metadatas": [[{"source": "k8s/deployments.md", "repo": "kubernetes/website"}]],
        "distances": [[0.08]],
    }

    with patch(
        "app.services.rag_service.open_persistent_collection",
        return_value=fake_collection,
    ) as opener:
        service = RAGService(settings)
        service.connect()
        import asyncio

        chunks = asyncio.run(service.query_context("rolling update", top_k=3))

    opener.assert_called_once()
    assert opener.call_args.kwargs["collection_name"] == "tech_tutor_corpus"
    assert len(chunks) == 1
    assert chunks[0].source == "k8s/deployments.md"
    assert chunks[0].distance == 0.08
    assert "maxSurge" in chunks[0].document


def test_singleton_identity(tmp_path: Path) -> None:
    persist = tmp_path / "db"
    persist.mkdir()
    settings = Settings(chroma_persist_dir=str(persist))
    first = RAGService(settings)
    second = RAGService(settings)
    assert first is second
