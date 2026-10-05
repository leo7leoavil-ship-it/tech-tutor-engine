"""POST /api/v1/evaluate/voice — ASR + RAG + Llama 3 evaluation."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.core.config import get_settings
from app.core.security import verify_api_key
from app.schemas.response_schemas import EvaluationResponse
from app.services.pipeline import run_voice_evaluation

router = APIRouter()

_ALLOWED_CONTENT_TYPES = {
    "audio/webm",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "application/octet-stream",
}
_ALLOWED_SUFFIXES = {".webm", ".mp3", ".wav", ".ogg", ".m4a"}


@router.post(
    "/evaluate/voice",
    response_model=EvaluationResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["evaluation"],
)
async def evaluate_voice(
    request: Request,
    audio_file: UploadFile = File(..., description="Spoken answer as .webm or .mp3"),
    target_scenario: str = Form("system_design_interview"),
) -> EvaluationResponse:
    """Orchestrate Audio Bytes → ASR → RAG → LLM structured feedback.

    Args:
        audio_file: In-memory uploaded container (WebM / MP3).
        target_scenario: Interview scenario that conditions the tutor prompt.

    Returns:
        EvaluationResponse including `execution_time_ms`.
    """
    _validate_audio_upload(audio_file)
    audio_bytes = await audio_file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    settings = get_settings()
    try:
        return await run_voice_evaluation(
            audio_bytes=audio_bytes,
            target_scenario=target_scenario,
            asr=request.app.state.asr,
            rag=request.app.state.rag,
            llm=request.app.state.llm,
            top_k=settings.rag_top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def _validate_audio_upload(audio_file: UploadFile) -> None:
    """Reject obviously non-audio uploads before decoding."""
    filename = (audio_file.filename or "").lower()
    content_type = (audio_file.content_type or "").lower()
    suffix_ok = any(filename.endswith(suffix) for suffix in _ALLOWED_SUFFIXES)
    type_ok = content_type in _ALLOWED_CONTENT_TYPES or content_type.startswith("audio/")
    if filename and not suffix_ok and not type_ok:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio type. Use .webm or .mp3.",
        )
