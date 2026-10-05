"""Llama 3 reasoning core with RAG-grounded structured JSON output."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.response_schemas import (
    EvaluationCore,
    FeedbackDetail,
    LLMReasoningOutput,
    RetrievedChunk,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Senior AI Technical Recruiter and English tutor for software,
platform, and infrastructure engineers. Evaluate the candidate's spoken technical English.

Hard rules:
- Use ONLY the Ground Truth context from the technical corpus to judge domain accuracy.
- Never invent APIs, protocols, SRE practices, or product names that are not supported
  by the retrieved context or by widely established public knowledge already restated
  in that context.
- Feedback must be specific, professional, and actionable for engineers.
- Output MUST be valid JSON matching this schema (no markdown, no prose outside JSON):
{
  "evaluation_core": {
    "overall_score": <float 0-100>,
    "lexical_density_ratio": <float 0-1>,
    "grammar_accuracy_score": <float 0-100>,
    "detected_technical_terms": [<string>, ...]
  },
  "feedback_details": [
    {
      "type": "<grammar|lexicon|discourse|technical_accuracy>",
      "original_phrase": "<from transcript>",
      "suggested_phrase": "<idiomatic rewrite>",
      "rag_source_context": "<short quote from ground truth>",
      "explanation": "<why this matters in a tech interview>"
    }
  ]
}
If the transcript is empty, score overall_score as 0 and explain that no speech was captured.
"""

HUMAN_PROMPT = """Target interview scenario: {target_scenario}

ASR transcript:
\"\"\"{transcript}\"\"\"

Ground Truth context (ChromaDB tech_tutor_corpus, ranked by cosine similarity):
{rag_context}

Produce the JSON evaluation now.
"""


class LLMService:
    """Llama 3 via an OpenAI-compatible chat API (Groq / HF / Together).

    Structured output is enforced with LangChain `with_structured_output`
    against `LLMReasoningOutput`. Timing, ASR metrics, and HTTP envelope
    fields are assembled by the API orchestrator, not by the model.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm: BaseChatModel | None = None
        self._structured = None
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ]
        )

    def load(self) -> None:
        """Instantiate the chat client. Does not perform a network round-trip."""
        api_key = self._settings.llm_api_key.get_secret_value()
        if not api_key:
            logger.warning("LLM_API_KEY is empty; evaluate endpoints will fail until configured.")

        self._llm = ChatOpenAI(
            model=self._settings.llm_model,
            api_key=api_key or "missing-key",
            base_url=self._settings.llm_base_url,
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            timeout=self._settings.llm_timeout_seconds,
        )
        self._structured = self._llm.with_structured_output(
            LLMReasoningOutput,
            method="json_mode",
        )
        logger.info(
            "LLM client ready provider=%s model=%s base_url=%s",
            self._settings.llm_provider,
            self._settings.llm_model,
            self._settings.llm_base_url,
        )

    @property
    def is_configured(self) -> bool:
        """Whether an API key is present (not a live health ping)."""
        return bool(self._settings.llm_api_key.get_secret_value())

    async def evaluate(
        self,
        transcript: str,
        rag_chunks: list[RetrievedChunk],
        target_scenario: str,
    ) -> LLMReasoningOutput:
        """Run RAG-grounded evaluation and parse into `LLMReasoningOutput`.

        Args:
            transcript: ASR text.
            rag_chunks: Retrieved corpus neighbours.
            target_scenario: Interview scenario key from the request form.

        Returns:
            Validated reasoning payload. Falls back to a conservative
            heuristic object if the provider returns unparseable content.
        """
        if self._structured is None or self._llm is None:
            raise RuntimeError("LLMService.load() must be called during application startup.")

        rag_context = _format_rag_context(rag_chunks)
        chain = self._prompt | self._structured
        try:
            result = await chain.ainvoke(
                {
                    "target_scenario": target_scenario,
                    "transcript": transcript or "(empty transcript)",
                    "rag_context": rag_context,
                }
            )
            if isinstance(result, LLMReasoningOutput):
                return result
            if isinstance(result, dict):
                return LLMReasoningOutput.model_validate(result)
            return _parse_loose_json(str(result))
        except (ValidationError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Structured output failed (%s); using json-mode fallback.", exc)
            return await self._json_mode_fallback(transcript, rag_context, target_scenario)
        except Exception:
            logger.exception("LLM provider call failed")
            raise

    async def _json_mode_fallback(
        self,
        transcript: str,
        rag_context: str,
        target_scenario: str,
    ) -> LLMReasoningOutput:
        """Retry with JSON-mode chat completion and manual Pydantic parse."""
        assert self._llm is not None
        messages = self._prompt.format_messages(
            target_scenario=target_scenario,
            transcript=transcript or "(empty transcript)",
            rag_context=rag_context,
        )
        raw = await self._llm.ainvoke(messages)
        content = raw.content if isinstance(raw.content, str) else json.dumps(raw.content)
        return _parse_loose_json(content)


def _format_rag_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved neighbours as numbered ground-truth blocks."""
    if not chunks:
        return "(no corpus neighbours retrieved)"
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.source or "unknown_source"
        distance = f"{chunk.distance:.4f}" if chunk.distance is not None else "n/a"
        blocks.append(
            f"[{index}] source={source} cosine_distance={distance}\n{chunk.document}"
        )
    return "\n\n".join(blocks)


def _parse_loose_json(content: str) -> LLMReasoningOutput:
    """Strip optional markdown fences and validate against the schema."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        return _empty_reasoning("Model returned non-JSON content.")
    payload = json.loads(stripped[start : end + 1])
    try:
        return LLMReasoningOutput.model_validate(payload)
    except ValidationError:
        return _empty_reasoning("Model JSON did not match EvaluationResponse subset.")


def _empty_reasoning(explanation: str) -> LLMReasoningOutput:
    """Deterministic fallback so the API still returns a valid envelope."""
    return LLMReasoningOutput(
        evaluation_core=EvaluationCore(
            overall_score=0.0,
            lexical_density_ratio=0.0,
            grammar_accuracy_score=0.0,
            detected_technical_terms=[],
        ),
        feedback_details=[
            FeedbackDetail(
                type="technical_accuracy",
                original_phrase="",
                suggested_phrase="",
                rag_source_context="",
                explanation=explanation,
            )
        ],
    )
