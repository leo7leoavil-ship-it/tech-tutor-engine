"""Pydantic v2 response models for structured tutor feedback."""

from typing import Literal

from pydantic import BaseModel, Field


class ASRMetrics(BaseModel):
    """Automatic Speech Recognition quality snapshot."""

    transcription: str = Field(..., description="Decoded utterance in English.")
    detected_language: str = Field(..., description="ISO-639 language code from Whisper.")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate ASR confidence in [0, 1].",
    )


class EvaluationCore(BaseModel):
    """Quantitative scores produced by the Llama 3 reasoning core."""

    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite technical-English score on a 0–100 scale.",
    )
    lexical_density_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Content-word ratio (technical lexis vs. filler).",
    )
    grammar_accuracy_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Grammar accuracy on a 0–100 scale.",
    )
    detected_technical_terms: list[str] = Field(
        default_factory=list,
        description="Domain terms identified in the utterance.",
    )


class FeedbackDetail(BaseModel):
    """Single actionable correction grounded in RAG context."""

    type: str = Field(
        ...,
        description="Feedback category, e.g. grammar, lexicon, pronunciation_proxy, technical_accuracy.",
        examples=["grammar"],
    )
    original_phrase: str = Field(..., description="Fragment taken from the ASR transcript.")
    suggested_phrase: str = Field(..., description="Idiomatic / technically accurate rewrite.")
    rag_source_context: str = Field(
        ...,
        description="Ground-truth excerpt retrieved from the technical corpus.",
    )
    explanation: str = Field(..., description="Tutor explanation in English for engineers.")


class EvaluationResponse(BaseModel):
    """Canonical structured payload returned by `/evaluate/voice` and the WS channel."""

    status: Literal["success"] = "success"
    execution_time_ms: float = Field(..., ge=0.0, description="End-to-end pipeline latency.")
    asr_metrics: ASRMetrics
    evaluation_core: EvaluationCore
    feedback_details: list[FeedbackDetail] = Field(default_factory=list)


class LLMReasoningOutput(BaseModel):
    """Subset of EvaluationResponse that the LLM is allowed to generate."""

    evaluation_core: EvaluationCore
    feedback_details: list[FeedbackDetail] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """Single ChromaDB neighbour with source traceability."""

    document: str
    metadata: dict[str, str | int | float | None] = Field(default_factory=dict)
    distance: float | None = Field(
        default=None,
        description="Cosine distance from the query embedding (lower is closer).",
    )
    source: str | None = Field(
        default=None,
        description="Origin repository / document path from chunk metadata.",
    )


class QueryResponse(BaseModel):
    """Text-only RAG retrieval result."""

    status: Literal["success"] = "success"
    query_text: str
    top_k: int
    chunks: list[RetrievedChunk]


class HealthResponse(BaseModel):
    """Readiness probe payload."""

    status: Literal["ok", "degraded"] = "ok"
    app: str
    version: str
    chroma_ready: bool
    asr_ready: bool
    llm_configured: bool
