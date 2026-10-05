"""Pydantic request models for REST and WebSocket payloads."""

from typing import Literal

from pydantic import BaseModel, Field

TargetScenario = Literal[
    "system_design_interview",
    "code_review",
    "incident_postmortem",
    "architecture_walkthrough",
    "technical_standup",
]


class QueryRequest(BaseModel):
    """Text-only RAG retrieval probe (no ASR / LLM evaluation)."""

    query_text: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="Engineer utterance or interview prompt to retrieve context for.",
        examples=["Explain how Kubernetes rolling updates avoid downtime."],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of cosine-similarity neighbours to return.",
    )


class PracticeControlMessage(BaseModel):
    """JSON control frame on the practice WebSocket."""

    event: Literal["evaluate", "reset", "ping"] = "evaluate"
    target_scenario: TargetScenario = "system_design_interview"
