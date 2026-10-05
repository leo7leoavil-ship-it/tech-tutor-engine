"""Whisper Large-v3-Turbo ASR wrapper over in-memory audio buffers."""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.schemas.response_schemas import ASRMetrics

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ASRService:
    """Asynchronous speech-to-text using faster-whisper (CTranslate2).

    Audio is decoded from an in-memory WebM/MP3/WAV buffer via PyAV — no
    persistent disk writes. Model load is performed once at process start.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Load Whisper Large-v3-Turbo into memory (blocking, call at startup)."""
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s",
            self._settings.whisper_model_size,
            self._settings.whisper_device,
            self._settings.whisper_compute_type,
        )
        self._model = WhisperModel(
            self._settings.whisper_model_size,
            device=self._settings.whisper_device,
            compute_type=self._settings.whisper_compute_type,
        )

    @property
    def is_ready(self) -> bool:
        """Whether the CTranslate2 model weights are resident."""
        return self._model is not None

    async def transcribe(self, audio_bytes: bytes) -> ASRMetrics:
        """Transcribe an in-memory audio buffer.

        Args:
            audio_bytes: Raw WebM, MP3, WAV, or OGG container bytes.

        Returns:
            ASRMetrics with transcript, language, and mean segment confidence.

        Raises:
            RuntimeError: If the model was not loaded or audio is empty.
            ValueError: If the buffer cannot be decoded.
        """
        if not audio_bytes:
            raise ValueError("Empty audio buffer.")
        if self._model is None:
            raise RuntimeError("ASRService.load() must be called during application startup.")

        return await asyncio.to_thread(self._transcribe_sync, audio_bytes)

    def _transcribe_sync(self, audio_bytes: bytes) -> ASRMetrics:
        """Blocking decode + inference executed on a worker thread."""
        assert self._model is not None
        try:
            waveform = decode_audio_buffer(audio_bytes)
        except Exception as exc:  # noqa: BLE001 — PyAV raises heterogeneous types
            logger.exception("Failed to decode in-memory audio buffer")
            raise ValueError("Unsupported or corrupted audio buffer.") from exc

        language = self._settings.whisper_language or None
        segments_iter, info = self._model.transcribe(
            waveform,
            beam_size=self._settings.whisper_beam_size,
            language=language,
            vad_filter=True,
        )
        segments = list(segments_iter)
        transcription = " ".join(segment.text.strip() for segment in segments).strip()
        confidences = [
            segment.avg_logprob for segment in segments if segment.avg_logprob is not None
        ]
        # avg_logprob is typically in [-1, 0]; map to a bounded [0, 1] confidence.
        if confidences:
            mean_logprob = sum(confidences) / len(confidences)
            confidence = max(0.0, min(1.0, 1.0 + mean_logprob))
        else:
            confidence = 0.0

        detected = info.language or language or "en"
        logger.debug(
            "ASR done language=%s duration=%.2fs confidence=%.3f",
            detected,
            getattr(info, "duration", 0.0),
            confidence,
        )
        return ASRMetrics(
            transcription=transcription,
            detected_language=detected,
            confidence_score=round(confidence, 4),
        )


def decode_audio_buffer(audio_bytes: bytes):
    """Decode a container from memory to a 16 kHz mono waveform."""
    from faster_whisper.audio import decode_audio

    return decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000)
