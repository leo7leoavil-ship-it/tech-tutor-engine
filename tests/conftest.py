"""Shared pytest fixtures. Heavy models are never loaded in unit tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

os.environ["SKIP_HEAVY_INIT"] = "true"
os.environ["API_KEY"] = "test-api-key"
os.environ["LLM_API_KEY"] = "test-llm-key"
os.environ["CHROMA_PERSIST_DIR"] = "./data/technical_corpus_db"

from app.core.config import get_settings  # noqa: E402
from app.schemas.response_schemas import (  # noqa: E402
    ASRMetrics,
    EvaluationCore,
    EvaluationResponse,
    FeedbackDetail,
    RetrievedChunk,
)

get_settings.cache_clear()

AUTH_HEADERS = {"Authorization": "Bearer test-api-key"}


def _sample_evaluation() -> EvaluationResponse:
    return EvaluationResponse(
        status="success",
        execution_time_ms=12.5,
        asr_metrics=ASRMetrics(
            transcription="Kubernetes rolling updates keep a surge of new pods.",
            detected_language="en",
            confidence_score=0.91,
        ),
        evaluation_core=EvaluationCore(
            overall_score=82.0,
            lexical_density_ratio=0.61,
            grammar_accuracy_score=88.0,
            detected_technical_terms=["Kubernetes", "rolling update"],
        ),
        feedback_details=[
            FeedbackDetail(
                type="lexicon",
                original_phrase="surge of new pods",
                suggested_phrase="surge of new ReplicaSet pods",
                rag_source_context="RollingUpdate maxSurge controls extra pods.",
                explanation="Prefer the precise Deployment strategy term maxSurge.",
            )
        ],
    )


@pytest.fixture
def mock_services() -> dict[str, MagicMock]:
    """In-memory stand-ins for ASR, RAG, and LLM."""
    rag = MagicMock()
    rag.is_ready = True
    rag.query_context = AsyncMock(
        return_value=[
            RetrievedChunk(
                document="RollingUpdate maxSurge controls extra pods during a rollout.",
                metadata={"source": "kubernetes/docs/deployments.md"},
                distance=0.12,
                source="kubernetes/docs/deployments.md",
            )
        ]
    )

    asr = MagicMock()
    asr.is_ready = True
    asr.transcribe = AsyncMock(
        return_value=ASRMetrics(
            transcription="Kubernetes rolling updates keep a surge of new pods.",
            detected_language="en",
            confidence_score=0.91,
        )
    )

    llm = MagicMock()
    llm.is_configured = True
    llm.is_ready = True
    sample = _sample_evaluation()
    llm.evaluate = AsyncMock(
        return_value=MagicMock(
            evaluation_core=sample.evaluation_core,
            feedback_details=sample.feedback_details,
        )
    )
    return {"rag": rag, "asr": asr, "llm": llm}


@pytest.fixture
def client(mock_services: dict[str, MagicMock]) -> Iterator[TestClient]:
    """HTTP client with lifespan skipped and services injected."""
    from app.main import app

    app.state.rag = mock_services["rag"]
    app.state.asr = mock_services["asr"]
    app.state.llm = mock_services["llm"]
    with TestClient(app) as test_client:
        app.state.rag = mock_services["rag"]
        app.state.asr = mock_services["asr"]
        app.state.llm = mock_services["llm"]
        yield test_client
