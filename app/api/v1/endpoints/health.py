"""GET /health — process readiness (Chroma, ASR load, LLM key)."""

from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.schemas.response_schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(request: Request) -> HealthResponse:
    """Readiness probe for Hugging Face Spaces and orchestrators.

    Reports whether the persistent Chroma collection is bound, Whisper is
    resident, and an LLM API key is configured. Does not call the LLM provider.
    """
    settings = get_settings()
    rag = getattr(request.app.state, "rag", None)
    asr = getattr(request.app.state, "asr", None)
    llm = getattr(request.app.state, "llm", None)

    chroma_ready = bool(rag and rag.is_ready)
    asr_ready = bool(asr and asr.is_ready)
    llm_configured = bool(llm and llm.is_configured)
    ok = chroma_ready and asr_ready and llm_configured
    return HealthResponse(
        status="ok" if ok else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        chroma_ready=chroma_ready,
        asr_ready=asr_ready,
        llm_configured=llm_configured,
    )
