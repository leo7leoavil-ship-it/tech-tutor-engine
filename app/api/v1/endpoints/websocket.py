"""WS /api/v1/ws/practice — real-time audio streaming for mock interviews."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.security import verify_ws_token
from app.schemas.request_schemas import PracticeControlMessage
from app.services.pipeline import run_voice_evaluation

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/practice")
async def practice_socket(
    websocket: WebSocket,
    token: str = Depends(verify_ws_token),
) -> None:
    """Bidirectional practice channel.

    Protocol:
        1. Client connects to `/api/v1/ws/practice?token=<API_KEY>`.
        2. Client streams binary WebM/MP3 frames (appended in-memory).
        3. Client sends a JSON text frame `{"event":"evaluate","target_scenario":"..."}`.
        4. Server runs ASR → RAG → LLM and replies with `EvaluationResponse` JSON.
        5. `{"event":"reset"}` clears the audio buffer; `{"event":"ping"}` → `{"event":"pong"}`.

    Args:
        websocket: FastAPI WebSocket.
        token: Validated API key (query parameter).
    """
    del token  # authenticated by dependency
    await websocket.accept()
    settings = get_settings()
    buffer = bytearray()
    await websocket.send_json(
        {
            "event": "ready",
            "max_audio_bytes": settings.ws_max_audio_bytes,
            "hint": "Send binary audio frames, then a JSON evaluate event.",
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is not None:
                if len(buffer) + len(data) > settings.ws_max_audio_bytes:
                    await websocket.send_json(
                        {
                            "event": "error",
                            "detail": "Audio buffer exceeded WS_MAX_AUDIO_BYTES.",
                        }
                    )
                    buffer.clear()
                    continue
                buffer.extend(data)
                await websocket.send_json(
                    {"event": "ack", "buffered_bytes": len(buffer)}
                )
                continue

            text = message.get("text")
            if text is None:
                continue
            await _handle_control_frame(websocket, text, buffer)
    except WebSocketDisconnect:
        logger.info("Practice WebSocket disconnected")
    except Exception:
        logger.exception("Practice WebSocket failed")
        await _safe_close(websocket)


async def _handle_control_frame(
    websocket: WebSocket,
    text: str,
    buffer: bytearray,
) -> None:
    """Dispatch a JSON control message against the in-memory audio buffer."""
    try:
        control = PracticeControlMessage.model_validate_json(text)
    except ValidationError as exc:
        await websocket.send_json({"event": "error", "detail": json.loads(exc.json())})
        return

    if control.event == "ping":
        await websocket.send_json({"event": "pong"})
        return

    if control.event == "reset":
        buffer.clear()
        await websocket.send_json({"event": "reset", "buffered_bytes": 0})
        return

    if control.event == "evaluate":
        if not buffer:
            await websocket.send_json(
                {"event": "error", "detail": "No audio buffered. Send binary frames first."}
            )
            return
        settings = get_settings()
        try:
            result = await run_voice_evaluation(
                audio_bytes=bytes(buffer),
                target_scenario=control.target_scenario,
                asr=websocket.app.state.asr,
                rag=websocket.app.state.rag,
                llm=websocket.app.state.llm,
                top_k=settings.rag_top_k,
            )
        except ValueError as exc:
            await websocket.send_json({"event": "error", "detail": str(exc)})
            return
        buffer.clear()
        await websocket.send_json(
            {"event": "evaluation", "payload": result.model_dump()}
        )


async def _safe_close(websocket: WebSocket) -> None:
    """Close the socket if the peer is still connected."""
    try:
        await websocket.close(code=1011)
    except RuntimeError:
        return
