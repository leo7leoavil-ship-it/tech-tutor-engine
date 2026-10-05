"""Unit tests for the faster-whisper ASR wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import get_settings
from app.services.asr_service import ASRService


class TestASRService:
    """In-memory transcription without disk I/O."""

    def test_transcribe_rejects_empty_buffer(self) -> None:
        service = ASRService(get_settings())
        service._model = MagicMock()
        with pytest.raises(ValueError, match="Empty audio"):
            # asyncio.run would also work; pytest-asyncio fixture not required here
            import asyncio

            asyncio.run(service.transcribe(b""))

    def test_transcribe_requires_loaded_model(self) -> None:
        service = ASRService(get_settings())
        import asyncio

        with pytest.raises(RuntimeError, match="load"):
            asyncio.run(service.transcribe(b"fake-bytes"))

    def test_transcribe_maps_logprob_to_confidence(self) -> None:
        service = ASRService(get_settings())
        service._model = MagicMock()
        segment = SimpleNamespace(text="  Rolling updates  ", avg_logprob=-0.2)
        info = SimpleNamespace(language="en", duration=1.2)
        service._model.transcribe.return_value = ([segment], info)

        with patch(
            "app.services.asr_service.decode_audio_buffer",
            return_value=[0.0] * 16000,
        ):
            import asyncio

            metrics = asyncio.run(service.transcribe(b"ID3fakeaudio"))

        assert metrics.transcription == "Rolling updates"
        assert metrics.detected_language == "en"
        assert 0.7 <= metrics.confidence_score <= 0.9
