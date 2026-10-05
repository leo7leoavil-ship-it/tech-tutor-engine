"""End-to-end evaluation orchestrator: ASR → RAG → LLM."""

from __future__ import annotations

import time

from app.schemas.response_schemas import EvaluationResponse
from app.services.asr_service import ASRService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


async def run_voice_evaluation(
    audio_bytes: bytes,
    target_scenario: str,
    asr: ASRService,
    rag: RAGService,
    llm: LLMService,
    top_k: int = 3,
) -> EvaluationResponse:
    """Execute the production voice-evaluation pipeline.

    Args:
        audio_bytes: In-memory WebM/MP3/WAV payload.
        target_scenario: Interview scenario identifier.
        asr: Loaded Whisper service.
        rag: Connected ChromaDB service.
        llm: Loaded Llama 3 service.
        top_k: RAG neighbour count.

    Returns:
        Fully populated EvaluationResponse including wall-clock latency.
    """
    started = time.perf_counter()
    asr_metrics = await asr.transcribe(audio_bytes)
    chunks = await rag.query_context(asr_metrics.transcription, top_k=top_k)
    reasoning = await llm.evaluate(
        transcript=asr_metrics.transcription,
        rag_chunks=chunks,
        target_scenario=target_scenario,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return EvaluationResponse(
        status="success",
        execution_time_ms=round(elapsed_ms, 2),
        asr_metrics=asr_metrics,
        evaluation_core=reasoning.evaluation_core,
        feedback_details=reasoning.feedback_details,
    )
